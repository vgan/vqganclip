#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Licensed under the MIT License

# Copyright (c) 2021 Steve Cvar

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# This script is based on code originally created by Katherine Crowson and Sebastián Bórquez

"""
**Download pre-trained models:**
get_models.py (--list | --all | -d (<model name>)+)
"""
import os, random
from extra import *
from subprocess import Popen, PIPE
from datetime import datetime
strTime = datetime.now()
strTime = strTime.strftime('%Y%m%d%H%M%S')

def sinc(x):
    return torch.where(x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))

def lanczos(x, a):
    cond = torch.logical_and(-a < x, x < a)
    out = torch.where(cond, sinc(x) * sinc(x/a), x.new_zeros([]))
    return out / out.sum()

def ramp(ratio, width):
    n = math.ceil(width / ratio + 1)
    out = torch.empty([n])
    cur = 0
    for i in range(out.shape[0]):
        out[i] = cur
        cur += ratio
    return torch.cat([-out[1:].flip([0]), out])[1:-1]

def resample(input, size, align_corners=True):
    n, c, h, w = input.shape
    dh, dw = size

    input = input.view([n * c, 1, h, w])

    if dh < h:
        kernel_h = lanczos(ramp(dh / h, 2), 2).to(input.device, input.dtype)
        pad_h = (kernel_h.shape[0] - 1) // 2
        input = F.pad(input, (0, 0, pad_h, pad_h), 'reflect')
        input = F.conv2d(input, kernel_h[None, None, :, None])

    if dw < w:
        kernel_w = lanczos(ramp(dw / w, 2), 2).to(input.device, input.dtype)
        pad_w = (kernel_w.shape[0] - 1) // 2
        input = F.pad(input, (pad_w, pad_w, 0, 0), 'reflect')
        input = F.conv2d(input, kernel_w[None, None, None, :])

    input = input.view([n, c, h, w])
    return F.interpolate(input, size, mode='bicubic', align_corners=align_corners)


class ReplaceGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_forward, x_backward):
        ctx.shape = x_backward.shape
        return x_forward

    @staticmethod
    def backward(ctx, grad_in):
        return None, grad_in.sum_to_size(ctx.shape)


class ClampWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min, max):
        ctx.min = min
        ctx.max = max
        ctx.save_for_backward(input)
        return input.clamp(min, max)

    @staticmethod
    def backward(ctx, grad_in):
        input, = ctx.saved_tensors
        return grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0), None, None

def vector_quantize(x, codebook):
    d = x.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(dim=1) - 2 * x @ codebook.T
    indices = d.argmin(-1)
    x_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
    return ReplaceGrad.apply(x_q, x)

class Prompt(nn.Module):
    def __init__(self, embed, weight=1., stop=float('-inf')):
        super().__init__()
        self.register_buffer('embed', embed)
        self.register_buffer('weight', torch.as_tensor(weight))
        self.register_buffer('stop', torch.as_tensor(stop))

    def forward(self, input):
        input_normed = F.normalize(input.unsqueeze(1), dim=2)
        embed_normed = F.normalize(self.embed.unsqueeze(0), dim=2)
        dists = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
        dists = dists * self.weight.sign()
        return self.weight.abs() * ReplaceGrad.apply(dists, torch.maximum(dists, self.stop)).mean()


def parse_prompt(prompt):
    vals = prompt.rsplit(':', 2)
    vals = vals + ['', '1', '-inf'][len(vals):]
    return vals[0], float(vals[1]), float(vals[2])


class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow
        self.augs = nn.Sequential(
            K.RandomHorizontalFlip(p=0.5),
            K.ColorJitter(hue=0.01, saturation=0.01, p=0.7),
            # K.RandomSolarize(0.01, 0.01, p=0.7),
            K.RandomSharpness(0.3,p=0.4),
            K.RandomAffine(degrees=30, translate=0.1, p=0.8, padding_mode='border'),
            K.RandomPerspective(0.2,p=0.4),)
        self.noise_fac = 0.1


    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
        batch = self.augs(torch.cat(cutouts, dim=0))
        if self.noise_fac:
            facs = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_fac)
            batch = batch + facs * torch.randn_like(batch)
        return batch


def load_vqgan_model(config_path, checkpoint_path):
    config = OmegaConf.load(config_path)
    if config.model.target == 'taming.models.vqgan.VQModel':
        model = vqgan.VQModel(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
    elif config.model.target == 'taming.models.cond_transformer.Net2NetTransformer':
        parent_model = cond_transformer.Net2NetTransformer(**config.model.params)
        parent_model.eval().requires_grad_(False)
        parent_model.init_from_ckpt(checkpoint_path)
        model = parent_model.first_stage_model
    else:
        raise ValueError(f'unknown model type: {config.model.target}')
    del model.loss
    return model


def resize_image(image, out_size):
    ratio = image.size[0] / image.size[1]
    area = min(image.size[0] * image.size[1], out_size[0] * out_size[1])
    size = round((area * ratio)**0.5), round((area / ratio)**0.5)
    return image.resize(size, Image.LANCZOS)


def synth(z, model):
    z_q = vector_quantize(z.movedim(1, 3), model.quantize.embedding.weight).movedim(3, 1)
    return ClampWithGrad.apply(model.decode(z_q).add(1).div(2), 0, 1)

def to_experiment_name(prompts):
    return " ".join(prompts).strip().replace(".", " ")\
              .replace("-", " ").replace(",", " ")\
              .replace(" ", "_")

def to_text(prompts):
    return " ".join(prompts).strip().replace(".", " ")\
        .replace("-", " ").replace(",", " ")\
        .replace("|", ", ")

def generate_images(
        prompts, model, outputs_folder, models_folder, iterations=200, image_prompts=[], 
        noise_prompt_seeds=[], noise_prompt_weights=[], size=[640, 480],
        init_image=None, init_weight=0., clip_model='ViT-B/32', 
        step_size=0.1, cutn=64, cut_pow=1., display_freq=5, seed=None,
        overwrite=False
    ):
    model_name = model
    experiment_name = to_experiment_name(prompts) + "_" + strTime
    print(experiment_name)
    experiment_folder = Path(outputs_folder) / experiment_name
    os.makedirs(experiment_folder, exist_ok=overwrite)
    os.makedirs(experiment_folder / "steps", exist_ok=overwrite)

    vqgan_config = Path(models_folder)/f'{model_name}.yaml'
    vqgan_checkpoint = Path(models_folder)/f'{model_name}.ckpt'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    model = load_vqgan_model(vqgan_config, vqgan_checkpoint).to(device)
    perceptor = clip.load(clip_model, jit=False)[0].eval().requires_grad_(False).to(device)

    cut_size = perceptor.visual.input_resolution
    e_dim = model.quantize.e_dim
    f = 2**(model.decoder.num_resolutions - 1)
    make_cutouts = MakeCutouts(cut_size, cutn, cut_pow=cut_pow)
    n_toks = model.quantize.n_e
    toksX, toksY = size[0] // f, size[1] // f
    sideX, sideY = toksX * f, toksY * f
    z_min = model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
    z_max = model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]

    if seed is not None:
        torch.manual_seed(seed)

    if init_image:
        pil_image = Image.open(init_image).convert('RGB')
        pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
        z, *_ = model.encode(TF.to_tensor(pil_image).to(device).unsqueeze(0) * 2 - 1)
    else:
        one_hot = F.one_hot(torch.randint(n_toks, [toksY * toksX], device=device), n_toks).float()
        z = one_hot @ model.quantize.embedding.weight
        z = z.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2)
    z_orig = z.clone()
    z.requires_grad_(True)
    opt = optim.Adam([z], lr=step_size)

    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                    std=[0.26862954, 0.26130258, 0.27577711])

    pMs = []

    for prompt in prompts:
        txt, weight, stop = parse_prompt(prompt)
        embed = perceptor.encode_text(clip.tokenize(txt).to(device)).float()
        pMs.append(Prompt(embed, weight, stop).to(device))

    for prompt in image_prompts:
        path, weight, stop = parse_prompt(prompt)
        img = resize_image(Image.open(path).convert('RGB'), (sideX, sideY))
        batch = make_cutouts(TF.to_tensor(img).unsqueeze(0).to(device))
        embed = perceptor.encode_image(normalize(batch)).float()
        pMs.append(Prompt(embed, weight, stop).to(device))

    for seed, weight in zip(noise_prompt_seeds, noise_prompt_weights):
        gen = torch.Generator().manual_seed(seed)
        embed = torch.empty([1, perceptor.visual.output_dim]).normal_(generator=gen)
        pMs.append(Prompt(embed, weight).to(device))

    @torch.no_grad()
    def checkin(z, i, losses, iterations, experiment_folder):
        out = synth(z, model)
        TF.to_pil_image(out[0].cpu()).save(experiment_folder / 'progress.png')

    def ascend_txt(pMs, z, i, init_weight, experiment_folder):
        out = synth(z, model)
        iii = perceptor.encode_image(normalize(make_cutouts(out))).float()
        result = []
        if init_weight:
            result.append(F.mse_loss(z, z_orig) * init_weight / 2)

        for prompt in pMs:
            result.append(prompt(iii))
        img = np.array(out.mul(255).clamp(0, 255)[0].cpu().detach().numpy().astype(np.uint8))[:,:,:]
        img = np.transpose(img, (1, 2, 0))
        imageio.imwrite(experiment_folder / 'steps'/  f"{str(i).zfill(3)}.png", np.array(img))
        return result

    def train(pMs, z, i, init_weight, display_freq, iterations, experiment_folder):
        opt.zero_grad()
        lossAll = ascend_txt(pMs, z, i, init_weight, experiment_folder)
        if i % display_freq == 0: checkin(z, i, lossAll, iterations, experiment_folder)
        loss = sum(lossAll)
        loss.backward()
        opt.step()
        with torch.no_grad():
            z.copy_(z.maximum(z_min).minimum(z_max))
    try:
        for i in tqdm(range(iterations), total=iterations, desc="Training"):
            train(pMs, z, i, init_weight, display_freq, iterations, experiment_folder)
    except KeyboardInterrupt:
        print("Aborted")

    image = str('/tf/outputs/' + experiment_name + '/progress.png')
    text = to_text(prompts)
    sendToInterwebs(image,text)
    return i

def generateText():
    style = random.choice([
        "a charcoal drawing",
        "a line drawing",
        "a vintage black velvet painting",
        "a vintage oil painting",
        "a watercolor painting",
        "Abstract",
        "Anime",
        "Art Deco",
        "Art Nouveau",
        "Aztec",
        "Baroque",
        "Constructivism",
        "Cubism",
        "Cyberpunk",
        "Dadaism",
        "Dali",
        "Expressionism",
        "Figurative art",
        "Geometric",
        "Gothic",
        "H.R. Giger",
        "Hieronymus Bosch",
        "Impressionism",
        "Matisse",
        "Mayan",
        "Michelangelo",
        "minimalist",
        "Moebius",
        "Moebius",
        "Moebius",
        "Moebius",
        "Moebius",
        "Monet",
        "Picasso",
        "Pop Art",
        "Portraiture",
        "Realism",
        "Retro-Futurism",
        "Roger Dean",
        "Social Realism",
        "Socialist Realism",
        "Still Life",
        "Surrealist",
        "Symbolism",
        "Trending on Art Station",
        "Ukiyo-E",
        "Van Gogh",
        "vaporwave",
        ])

    subject = random.choice([
        "a bear",
        "a blacksmith",
        "a cat",
        "a clown",
        "a cow",
        "a cowboy",
        "a crow",
        "a dragon",
        "a frog",
        "a ghost",
        "a hammer",
        "a hamster",
        "a high speed car chase",
        "a little girl",
        "a magician",
        "a man",
        "a moose",
        "a mouse",
        "a Muppet",
        "a parade",
        "a robot",
        "a space station",
        "a squirrel",
        "a sword",
        "a tardigrade",
        "a vacuum",
        "a volcano",
        "a wizard",
        "a wizard pondering an orb",
        "album cover",
        "an alien",
        "an elf",
        "angels",
        "bank robbers",
        "bees",
        "beetles",
        "bones",
        "bookshelves",
        "bridges",
        "camels",
        "carnivorous plants",
        "cats",
        "cavemen",
        "chupacabra",
        "clocks",
        "coffins",
        "deep sea fish",
        "demons",
        "detectives ",
        "dogs",
        "dumptrucks",
        "elves",
        "elvis",
        "fancy jewelry",
        "ferns",
        "ghosts",
        "giraffes",
        "goblins",
        "god",
        "godzilla",
        "hackers",
        "handcuffs",
        "jellyfish",
        "lizard men",
        "luigi",
        "lobsters",
        "mad scientists",
        "mariachis",
        "monsters",
        "muffins",
        "mutants",
        "nuclear explosion",
        "pancakes",
        "Pyramids",
        "raccoons",
        "satan",
        "scarabs",
        "scorpions",
        "sea shells",
        "sharks",
        "skiers",
        "skunks",
        "smartphones",
        "snakes"
        "Snoopy",
        "Super Mario",
        "sushi",
        "tacos",
        "teenagers",
        "the devil",
        "the pope",
        "tigers",
        "time travelers",
        "tombstones",
        "tractors",
        "trains",
        "Transformers",
        "turkeys",
        "UFO abduction",
        "UFOs",
        "umbrellas",
        "a Wienermobile",
        "wario"
        ])

    modifier = random.choice([
        "|abandoned",
        "|after the rain",
        "|autumn",
        "|black and white",
        "|blue tint",
        "|crystal matrix",
        "|dark",
        "|dim",
        "|dry",
        "|earth tones",
        "|faded",
        "|fall",
        "|futuristic",
        "|glowing",
        "|green tint",
        "|heavy rain",
        "|high contrast",
        "|illuminated",
        "|morning",
        "|neon",
        "|night",
        "|pastels",
        "|shining",
        "|shiny",
        "|spooky",
        "|stained glass",
        "|sunrise",
        "|sunset",
        "|wet",
        "|winter",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        ""])

    scene = random.choice([
        "at a concert",
        "at a dinner party",
        "at a dive bar",
        "at a grassy field",
        "at a pachinko parlor",
        "at a pool party",
        "at the circus",
        "at the gates of heaven",
        "at the gates of hell",
        "at the gates of valhalla",
        "at the great pyramids",
        "at the mall",
        "by a creek",
        "in a barn",
        "in a black hole",
        "in a castle",
        "in a cavern",
        "in a datacenter",
        "in a desert",
        "in a diner",
        "in a factory",
        "in a gothic cathedral",
        "in a graveyard",
        "in a hidden tunnel",
        "in a jungle",
        "in a medieval castle",
        "in a post-apocalyptic city",
        "in a secret passage",
        "in a swamp",
        "in an arcade",
        "in ancient Egypt",
        "in ancient Greece",
        "in ancient rome",
        "in Athens",
        "in Bangkok",
        "in Belgrade",
        "in Bishkek",
        "in Bombay",
        "in Bucharest",
        "in Buenos Aires",
        "in Caracas",
        "in Ethiopia",
        "in Havana",
        "in Helsinki",
        "in Hollywood",
        "in Istanbul",
        "in Krakow",
        "in Kyiv",
        "in Las Vegas",
        "in Medellin",
        "in Mexico City",
        "in Minsk",
        "in Montevideo",
        "in Odessa",
        "in Paris",
        "in Prague",
        "in Saigon",
        "in Samara Russia",
        "in Sao Paolo",
        "in Shanghai",
        "in Siberia",
        "in Sintra",
        "in Skopje",
        "in soviet russia",
        "in space",
        "in Taipei",
        "in Tallinn",
        "in Tashkent",
        "in Tbilisi",
        "in the city",
        "in the countryside",
        "in the metro",
        "in the mountains",
        "in the ocean",
        "in Tokyo",
        "in Ulaanbaatar",
        "inside a machine",
        "lakeside resort",
        "on a battleship",
        "on a cloud",
        "on a cruiseship",
        "on a farm",
        "on a pier",
        "on a ship",
        "on a train",
        "on a tropical island",
        "on an island",
        "on Mars",
        "on the beach",
        "on the Moon"
        ])
    place = random.choice([
        "a barn",
        "a black hole",
        "a castle",
        "a cavern",
        "a datacenter",
        "a desert",
        "a diner",
        "a dinner party",
        "a factory",
        "a Galaxy",
        "a gothic cathedral",
        "a grassy field",
        "a graveyard",
        "a hidden tunnel",
        "a jungle",
        "a medieval castle",
        "a pachinko parlor",
        "a pool party",
        "a post-apocalyptic city",
        "a secret passage",
        "a swamp",
        "an arcade",
        "ancient Egypt",
        "ancient Greece",
        "ancient rome",
        "Athens",
        "Bangkok",
        "Belgrade",
        "Bishkek",
        "Bombay",
        "Bucharest",
        "Buenos Aires",
        "by a creek",
        "Caracas",
        "Constructivist Architecture",
        "cyberspace",
        "Ethiopia",
        "Havana",
        "Helsinki",
        "Hollywood",
        "inside a machine",
        "Istanbul",
        "Krakow",
        "Kyiv",
        "lakeside resort",
        "Las Vegas",
        "Medellin",
        "Metabolist Architecture",
        "Mexico City",
        "Minsk",
        "Modernist Architecture",
        "Montevideo",
        "Odessa",
        "on a battleship",
        "on a cloud",
        "on a cruiseship",
        "on a farm",
        "on a pier",
        "on a ship",
        "on a train",
        "on a tropical island",
        "on an island",
        "on Mars",
        "on stage at a concert",
        "on the beach",
        "on the Moon",
        "Organic Brutalist Architecture",
        "Paris",
        "Postmodernist Architecture",
        "Prague",
        "Saigon",
        "Samara Russia",
        "Sao Paolo",
        "Shanghai",
        "Siberia",
        "Sintra",
        "Skopje",
        "soviet russia",
        "Taipei",
        "Tallinn",
        "Tangier",
        "Tashkent",
        "Tbisili",
        "the circus",
        "the city",
        "the countryside",
        "the gates of heaven",
        "the gates of hell",
        "the gates of valhalla",
        "the great pyramids",
        "the Interzone",
        "the mall",
        "the metro",
        "the mountains",
        "Tokyo",
        "Ulaanbaatar"
        ])

    preset_scene = random.choice([
        "1960s brutalist church interior",
        "1960s brutalist corridor",
        "1960s brutalist hotel",
        "1960s brutalist hotel |plants",
        "1960s modernist church interior",
        "1970s airbrushed tiger",
        "1970s dragon airbrushed",
        "1980s airbrushed fantasy movie poster",
        "1980s airbrushed galaxy",
        "1980s airbrushed knight",
        "1980s airbrushed movie poster",
        "1980s airbrushed space",
        "1990s taco bell dining area",
        "a space garden",
        "a surfer on a wave airbrushed",
        "album cover airbrushed",
        "animatronic abraham lincoln",
        "black metal album cover |evil",
        "Electron Microscopy dust mites",
        "Electron Microscopy spores",
        "Electron Microscopy tardigrade",
        "Electron Microscopy virus",
        "plants in a greenhouse at sunset",
        "unusual roadside attraction|japan|weird",
        "unusual roadside attraction|route 66",
        "unusual roadside attraction|weird"
        ])
    # define templates    
    sub_scene_style_mod = [subject,scene,"in the style of",style,modifier]
    sub_scene_style = [subject,scene,"in the style of",style]
    place_style_mod =[place,"in the style of",style,modifier]
    place_style = [place,"in the style of",style]
    preset = [preset_scene]

    template = random.choice([preset,sub_scene_style_mod,sub_scene_style,place_style_mod,place_style])

    text = template
    return text

def sendToInterwebs(image,text):
    # set keys from env variables
    client_key = os.environ['CLIENT_KEY']
    client_secret = os.environ['CLIENT_SECRET']
    access_token = os.environ['ACCESS_TOKEN']

    from mastodon import Mastodon
    mastodon = Mastodon(
        api_base_url='https://botsin.space',
        client_id=client_key,
        client_secret=client_secret,
        access_token = access_token
    )
    toot_text = ""
    description = text
    media_dict = mastodon.media_post(image,"image/png",description)
    mastodon.status_post(status=toot_text, media_ids=[media_dict,], sensitive=False)
    # Twitter
    import tweepy
    twitter_consumer_key = os.environ['TWITTER_CONSUMER_KEY']
    twitter_consumer_secret = os.environ['TWITTER_CONSUMER_SECRET']
    twitter_token_key = os.environ['TWITTER_TOKEN_KEY']
    twitter_token_secret = os.environ['TWITTER_TOKEN_SECRET']
    auth = tweepy.OAuthHandler(twitter_consumer_key, twitter_consumer_secret)
    auth.set_access_token(twitter_token_key, twitter_token_secret)
    api = tweepy.API(auth)
    imageObj = open(image)
    upload = api.media_upload(filename=image)
    media_ids = [upload.media_id_string]
    result = api.update_status(media_ids=media_ids, status="")

def doTheDamnedThing():
    prompts = generateText()
    model = "vqgan_imagenet_f16_16384"
    models_folder = "/tf/models"
    outputs_folder = "/tf/outputs"
    generate_images(prompts=prompts,model=model,outputs_folder=outputs_folder,models_folder=models_folder)

doTheDamnedThing()
