# %%
import os
os.environ["TRANSFORMERS_CACHE"] = "/workspace/cache/"
# %%
from transformers import LlamaForCausalLM, LlamaTokenizer
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-13b-hf")
tokenizer.pad_token_id = 0
tokenizer.bos_token_id = 1
tokenizer.eos_token_id = 2
# %%
from neel.imports import *
from neel_plotly import *

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.set_grad_enabled(False)

hf_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-chat-hf", torch_dtype=torch.float16)

cfg = loading.get_pretrained_model_config("llama-13b", torch_dtype=torch.float16)
print(cfg)
model = HookedTransformer(cfg, tokenizer=tokenizer)
state_dict = loading.get_pretrained_state_dict("llama-13b", cfg, hf_model, torch_dtype=torch.float16)
model.load_state_dict(state_dict, strict=False)


# model: HookedTransformer = HookedTransformer.from_pretrained_no_processing("llama-7b", hf_model=hf_model, tokenizer=tokenizer, device="cpu")
n_layers = model.cfg.n_layers
d_model = model.cfg.d_model
n_heads = model.cfg.n_heads
d_head = model.cfg.d_head
d_mlp = model.cfg.d_mlp
d_vocab = model.cfg.d_vocab
# %%
print(evals.sanity_check(model))
# %%
for layer in range(n_layers):
    model.blocks[layer].attn.W_K[:] = model.blocks[layer].attn.W_K * model.blocks[layer].ln1.w[None, :, None]
    model.blocks[layer].attn.W_Q[:] = model.blocks[layer].attn.W_Q * model.blocks[layer].ln1.w[None, :, None]
    model.blocks[layer].attn.W_V[:] = model.blocks[layer].attn.W_V * model.blocks[layer].ln1.w[None, :, None]
    model.blocks[layer].ln1.w[:] = torch.ones_like(model.blocks[layer].ln1.w)
    model.blocks[layer].mlp.W_in[:] = model.blocks[layer].mlp.W_in * model.blocks[layer].ln2.w[:, None]
    model.blocks[layer].mlp.W_gate[:] = model.blocks[layer].mlp.W_gate * model.blocks[layer].ln2.w[:, None]
    model.blocks[layer].ln2.w[:] = torch.ones_like(model.blocks[layer].ln2.w)
    
    model.blocks[layer].mlp.b_out[:] = model.blocks[layer].mlp.b_out + model.blocks[layer].mlp.b_in @ model.blocks[layer].mlp.W_out
    model.blocks[layer].mlp.b_in[:] = 0.

    model.blocks[layer].attn.b_O[:] = model.blocks[layer].attn.b_O[:] + (model.blocks[layer].attn.b_V[:, :, None] * model.blocks[layer].attn.W_O).sum([0, 1])
    model.blocks[layer].attn.b_V[:] = 0.

model.unembed.W_U[:] = model.unembed.W_U * model.ln_final.w[:, None]
model.unembed.W_U[:] = model.unembed.W_U - model.unembed.W_U.mean(-1, keepdim=True)
model.ln_final.w[:] = torch.ones_like(model.ln_final.w)
print(evals.sanity_check(model))
# %%
# %%
def decode_single_token(integer):
    # To recover whether the tokens begins with a space, we need to prepend a token to avoid weird start of string behaviour
    return tokenizer.decode([891, integer])[1:]
def to_str_tokens(tokens, prepend_bos=True):
    if isinstance(tokens, str):
        tokens = to_tokens(tokens)
    if isinstance(tokens, torch.Tensor):
        if len(tokens.shape)==2:
            assert tokens.shape[0]==1
            tokens = tokens[0]
        tokens = tokens.tolist()
    if prepend_bos:
        return [decode_single_token(token) for token in tokens]
    else:
        return [decode_single_token(token) for token in tokens[1:]]

def to_string(tokens):
    if isinstance(tokens, torch.Tensor):
        if len(tokens.shape)==2:
            assert tokens.shape[0]==1
            tokens = tokens[0]
        tokens = tokens.tolist()
    return tokenizer.decode([891]+tokens)[1:]
def to_tokens(string, prepend_bos=True):
    string = "|"+string
    # The first two elements are always [BOS (1), " |" (891)]
    tokens = tokenizer.encode(string)
    if prepend_bos:
        return torch.tensor(tokens[:1] + tokens[2:]).cuda()
    else:
        return torch.tensor(tokens[2:]).cuda()

def to_single_token(string):
    assert string[0]==" ", f"Expected string to start with space, got {string}"
    string = string[1:]
    tokens = tokenizer.encode(string)
    assert len(tokens)==2, f"Expected 2 tokens, got {len(tokens)}: {tokens}"
    return tokens[1]
print(to_str_tokens([270, 270]))
print(to_single_token(" basketball"))
# %%
# %%
# TF_TOKENS = [5852, 5574, 7700, 8824]
# TF_LABELS = [" True", "True", " False", "False"]
TF_TOKENS = [5852, 7700]
TF_LABELS = [" True", " False"]
logits = model("Answer with a single word. A human is larger than a cat. True or False?\n Answer:")
log_probs = logits[0, -1].log_softmax(-1)
print(log_probs[TF_TOKENS])
nutils.create_vocab_df(log_probs)
# %%
ENTITIES = list(set(["atom", "water molecule", "grain of sand", "pinhead", "flea", "pea", "marble", "mouse", "hamster", "cat", "dog", "soccer ball", "basketball", "beach ball", "human head", "bicycle", "motorcycle", "smart car", "sedan", "bus", "dump truck", "shipping container", "small house", "large house", "elephant", "blue whale", "passenger jet", "jumbo jet", "hot air balloon", "blimp", "massive cruise ship", "aircraft carrier", "football stadium", "asteroid", "moon", "cosmic microwave background radiation", "observable universe", "atom", "molecule", "virus", "bacteria", "pinhead", "ant", "mosquito", "housefly", "bee", "inchworm", "mouse", "rat", "guinea pig", "rabbit", "cat", "dog", "turkey", "alpaca", "donkey", "horse", "cow", "giraffe", "moose", "rhinoceros", "hippopotamus", "killer whale", "humpback whale", "giant squid", "great white shark", "octopus", "nautilus", "seahorse", "goldfish", "clownfish", "starfish", "abalone", "clam", "crab", "lobster", "shrimp", "jellyfish", "coral", "sea anemone", "plankton", "krill", "phytoplankton", "redwood tree", "oak tree", "banana tree", "bush", "shrub", "bamboo", "cactus", "orchid", "tulip", "sunflower", "mushroom", "apple", "orange", "coconut", "durian", "watermelon", "pumpkin", "pineapple", "avocado", "potato", "cabbage", "cauliflower", "broccoli", "zucchini", "bell pepper", "chili pepper", "fig", "grapes", "olive", "almond", "peanut", "walnut", "soybean", "lentil", "rice", "wheat", "corn", "pea", "marble", "golf ball", "baseball", "softball", "volleyball", "soccer ball", "football", "basketball", "bowling ball", "ping pong ball", "die", "domino", "playing card", "coin", "button", "rubber ducky", "doll", "action figure", "Lego brick", "toy car", "skateboard", "bicycle", "motorcycle", "scooter", "car", "van", "pick up truck", "semi truck", "bus", "tuk tuk", "fire truck", "ambulance", "police car", "tractor", "harvester", "bulldozer", "excavator", "crane", "forklift", "cement mixer", "Zamboni", "golf cart", "ATV", "dune buggy", "go kart", "jet ski", "speedboat", "yacht", "sailboat", "canoe", "kayak", "dinghy", "rowboat", "surfboard", "windsurfer", "hydrofoil", "cabin cruiser", "ferry", "river barge", "tugboat", "container ship", "oil tanker", "aircraft carrier", "submarine", "ice breaker", "cruise ship", "ocean liner", "space shuttle", "hot air balloon", "blimp", "biplane", "seaplane", "jumbo jet", "passenger jet", "fighter jet", "helicopter", "drone", "rocket", "space station", "booster rocket", "satellite", "Hubble space telescope", "space probe", "Apollo lunar module", "lunar rover", "Mars rover", "asteroid", "comet", "meteor", "dwarf planet", "moon", "planet", "star", "nebula", "galaxy", "galaxy cluster", "supercluster", "filament", "cosmic web", "observable universe", "quark", "electron", "proton", "neutron", "atom", "molecule", "buckyball", "nanotube", "virus", "bacteria", "cell", "multicellular organism", "ant", "flea", "mosquito", "fly", "bee", "beetle", "inchworm", "snail", "slug", "worm", "spider", "scorpion", "millipede", "centipede", "shrimp", "crab", "lobster", "clam", "oyster", "coral", "jellyfish", "starfish", "sea cucumber", "sea urchin", "plankton", "fish egg", "fish larva", "minnow", "goldfish", "tetra", "angelfish", "neon tetra", "guppy", "mollie", "betta", "catfish", "rainbow trout", "salmon", "tuna", "swordfish", "seahorse", "octopus", "squid", "nautilus", "sea slug", "sea snail", "limpet", "abalone", "sea cucumber", "sea urchin", "sand dollar", "mussel", "scallop", "oyster", "clam", "gecko", "skink", "chameleon", "iguana", "alligator", "crocodile", "turtle", "tortoise", "snake", "lizard", "salamander", "frog", "toad", "axolotl", "tadpole", "newt", "caecilian", "mouse", "rat", "vole", "lemming", "hamster", "gerbil", "guinea pig", "chinchilla", "hedgehog", "shrew", "mole", "pika", "rabbit", "hare", "capybara", "squirrel", "chipmunk", "marmot", "prairie dog", "porcupine", "beaver", "muskrat", "otter", "badger", "wolverine", "weasel", "ferret", "mink", "skunk", "raccoon", "red panda", "coati", "kinkajou", "sloth", "armadillo", "pangolin", "aardvark", "elephant shrew", "tenrec", "echidna", "platypus", "kangaroo", "koala", "wombat", "opossum", "Tasmanian devil", "numbat", "quoll", "bandicoot", "bilby", "sugar glider", "colugo", "treeshrew", "flying lemur", "flying fox", "fruit bat", "insectivorous bat", "vampire bat", "primate", "bushbaby", "loris", "tarsier", "lemur", "monkey", "ape", "gibbon", "orangutan", "gorilla", "chimpanzee", "bonobo", "baboon", "macaque", "vervet", "guenon", "patas monkey", "proboscis monkey", "colobus", "langur", "snub-nosed monkey", "howler monkey", "capuchin", "saki", "uakari", "titi", "spider monkey", "squirrel monkey", "marmoset", "tamarin", "night monkey", "owl monkey", "titi", "galago", "indri", "sifaka", "aye-aye", "humans", "pygmy mouse lemur", "mouse lemur", "dwarf lemur", "sportive lemur", "bamboo lemur", "brown lemur", "black lemur", "mongoose lemur", "ring-tailed lemur", "Grain of sand", "dust mite", "ant", "mosquito", "fly", "bee", "inchworm", "caterpillar", "snail", "beetle", "cricket", "worm", "spider", "ladybug", "pill bug", "centipede", "moth", "butterfly", "grasshopper", "frog", "lizard", "snake", "turtle", "hamster", "mouse", "rat", "squirrel", "rabbit", "cat", "dog", "chicken", "duck", "goose", "turkey", "pig", "sheep", "goat", "cow", "horse", "donkey", "llama", "fox", "raccoon", "skunk", "opossum", "woodchuck", "porcupine", "moose", "deer", "zebra", "rhino", "hippo", "elephant", "giraffe", "whale", "dolphin", "shark", "goldfish", "trout", "salmon", "tuna", "starfish", "coral", "sea urchin", "sponge", "jellyfish", "plankton", "leaf", "blade of grass", "flower", "bush", "fern", "cactus", "tree", "log", "stick", "pebble", "rock", "boulder", "brick", "book", "magazine", "newspaper", "envelope", "stamp", "coin", "button", "bead", "marble", "dice", "domino", "playing card", "eraser", "crayon", "pencil", "pen", "marker", "highlighter", "ruler", "scissors", "needle", "thimble", "push pin", "paper clip", "stapler", "tape", "glue stick", "rubber band", "binder clip", "paper weight", "hole punch", "staple remover", "calculator", "phone", "tablet", "laptop", "desktop", "printer", "keyboard", "mouse", "flash drive", "battery", "light bulb", "power cord", "extension cord", "surge protector", "clock", "watch", "hourglass", "compass", "telescope", "microscope", "stethoscope", "scalpel", "bandage", "crutches", "wheelchair", "car", "bus", "truck", "train", "plane", "helicopter", "rocket", "satellite", "space station", "bicycle", "motorcycle", "scooter", "skateboard", "roller skates", "ice skates", "skis", "snowboard", "surfboard", "kayak", "canoe", "sailboat", "yacht", "hot air balloon", "parade float", "house", "apartment", "hotel", "office building", "skyscraper", "bridge", "dam", "lighthouse", "windmill", "water tower", "pylon", "smokestack", "telescope", "satellite dish", "cell phone tower", "road", "highway", "sidewalk", "crosswalk", "parking lot", "driveway", "garage", "shed", "barn", "silo", "grain bin", "stable", "coop", "pen", "sty", "kennel", "doghouse", "birdhouse", "mailbox", "lamppost", "power pole", "fire hydrant", "stop sign", "yield sign", "traffic light", "bench", "bus stop sign", "parking meter", "statue", "fountain", "flag pole", "clothesline", "swing set", "slide", "seesaw", "sandbox", "grill", "lawn mower", "wheelbarrow", "shovel", "rake", "hoe", "sprinkler", "garden hose", "lawn chair", "picnic table", "umbrella", "tent", "camper", "RV", "trailer", "semi trailer", "shipping container", "dumpster", "landfill", "billboard", "storefront", "building", "house", "toolshed", "garage", "barn", "stadium", "skyscraper", "tower", "bridge", "dam", "lighthouse", "road", "path", "trail", "railroad", "runway", "sidewalk", "parking lot", "field", "farm", "pasture", "corral", "stable", "paddock", "meadow", "forest", "jungle", "swamp", "desert", "tundra", "glacier", "mountain", "volcano", "hill", "cliff", "canyon", "cave", "island", "lake", "pond", "river", "stream", "creek", "ocean", "sea", "bay", "gulf", "beach", "dune", "reef", "atoll", "iceberg", "floe", "ice sheet", "berg", "growler", "city", "town", "village", "neighborhood", "suburb", "metropolis", "state", "province", "county", "country", "continent", "hemisphere", "planet", "star", "galaxy", "universe", "grain of salt", "peppercorn", "poppy seed", "sesame seed", "kernel of corn", "grain of rice", "lentil", "pea", "bean", "pearl barley", "couscous", "quinoa", "pasta", "breadcrumb", "cookie crumb", "cracker crumb", "cereal flake", "chocolate chip", "raisin", "blueberry", "raspberry", "grape", "slice of apple", "orange segment", "mushroom cap", "brussel sprout", "pea pod", "grain of caviar", "slice of cheese", "pat of butter", "egg yolk", "bread slice", "pizza slice", "chicken nugget", "shrimp", "sushi roll", "dumpling", "ravioli", "gnocchi", "meatball", "chicken wing", "baby carrot", "grape tomato", "crouton", "tortilla chip", "pretzel", "potato chip", "French fry", "tater tot", "onion ring", "donut", "cupcake", "cookies", "measuring spoon", "fork", "knife", "chopstick", "spatula", "ladle", "tongs", "whisk", "spoon", "skewer", "grater", "peeler", "can opener", "corkscrew", "tenderizer", "timer", "strainer", "colander", "cheese grater", "zester", "masher", "meat mallet", "kitchen shears", "tongs", "spatula", "spatula", "wire whisk", "rolling pin", "measuring cups", "measuring spoons", "food scale", "mixing bowl", "saucepan", "stock pot", "skillet", "wok", "baking sheet", "baking dish", "casserole dish", "roasting pan", "muffin tin", "cake pan", "pie dish", "pizza stone", "pizza cutter", "pizza peel", "pot holder", "trivet", "blender", "food processor", "stand mixer", "hand mixer", "slow cooker", "pressure cooker", "rice cooker", "toaster", "toaster oven", "microwave", "oven", "stove", "range", "burner", "griddle", "broiler", "grill", "crockpot", "fridge", "freezer", "dishwasher", "sink", "kitchen island", "kitchen cart", "pantry", "cupboard", "cabinet", "drawer", "shelf", "dining table", "kitchen chair", "bar stool", "countertop", "backsplash", "floor tile", "wall tile", "light fixture", "faucet", "soap dispenser", "paper towel holder", "trash can", "compost bin", "vase", "candle", "kitchen utensil holder", "magnet", "photo", "artwork", "clock", "kitchen towel", "hot pad", "placemat", "napkin", "barn", "house", "cottage", "cabin", "hut", "yurt", "apartment", "condominium", "townhouse", "duplex", "triplex", "fourplex", "apartment building", "residential building", "skyscraper", "office building", "factory", "warehouse", "store", "shop", "mall", "garage", "hangar", "shed", "outbuilding", "gazebo", "pavilion", "stadium", "arena", "theater", "hall", "banquet hall", "place of worship", "church", "temple", "mosque", "synagogue", "chapel", "cathedral", "basilica", "courthouse", "city hall", "capital building", "state building", "parliament", "palace", "castle", "fortress", "market", "restaurant", "cafe", "hospital", "clinic", "doctor's office", "school", "university", "library", "post office", "fire station", "police station", "parking structure", "bus station", "train station", "airport", "spaceport", "pier", "dock", "lighthouse", "monument", "memorial", "mausoleum",]))
random.shuffle(ENTITIES)
train_entities = ENTITIES[:700]
valid_entities = ENTITIES[700:]
print(train_entities[:10])
print(valid_entities[:10])
# %%
# %%
def make_prompt(a, b):
    if a[0].lower() in "aeiou":
        prefix_a = "An"
    else:
        prefix_a = "A"
    if b[0].lower() in "aeiou":
        prefix_b = "an"
    else:
        prefix_b = "a"
    return f"Answer with a single word. {prefix_a} {a} is larger than {prefix_b} {b}. True or False?\n Answer:"

a = random.choice(ENTITIES)
b = random.choice(ENTITIES)
print(a, b)
tokens = torch.stack([to_tokens(make_prompt(a, b)), to_tokens(make_prompt(b, a))])
print(to_str_tokens(tokens[0]))
print(to_str_tokens(tokens[1]))
logits = model(tokens)
print(logits[:, -1, TF_TOKENS])
# %%
records = []
for i in tqdm.trange(10000):
    a = random.choice(ENTITIES)
    b = random.choice(ENTITIES)
    # print(a, b)
    tokens = torch.stack([to_tokens(make_prompt(a, b)), to_tokens(make_prompt(b, a))])
    # print(to_str_tokens(tokens[0]))
    # print(to_str_tokens(tokens[1]))
    logits = model(tokens)
    # logit_diff = logits[:, -1, TF_TOKENS[0]] - logits[:, -1, TF_TOKENS[1]]
    log_probs = to_numpy(logits[:, -1, :].log_softmax(dim=-1)[:, TF_TOKENS])
    records.append({
        "a":a,
        "b":b,
        "a>b_diff": log_probs[0, 0] - log_probs[0, 1],
        "b>a_diff": log_probs[1, 0] - log_probs[1, 1],
        "a>b_True": log_probs[0, 0],
        "b>a_True": log_probs[1, 0],
        "a>b_False": log_probs[0, 1],
        "b>a_False": log_probs[1, 1],
    })

# %%
df = pd.DataFrame(records)
# %%
df["consistent"] = (df["a>b_diff"]>0)==(df["b>a_diff"]<0)
df["consistent"].value_counts(True)
# %%
df["ave_diff"] = (df["a>b_diff"] - df["b>a_diff"])/2
df["ave_diff_abs"] = df["ave_diff"].abs()
sorted_df = df.sort_values("ave_diff_abs", ascending=False)
for i in range(0, 3000, 20):
    if sorted_df.ave_diff.iloc[i]>0:
        l1, l2 = "a", "b"
    else:
        l1, l2 = "b", "a"
    print(i, f"A {sorted_df.iloc[i][l1]} is bigger than a {sorted_df.iloc[i][l2]}", sorted_df.iloc[i]["ave_diff_abs"])
# %%
px.histogram(x=sorted_df["ave_diff"].to_numpy().astype(float), color=sorted_df["consistent"], barmode="overlay", marginal="box", histnorm="percent").show()
# %%
num_tokens_per_entity = {s: len(to_tokens(" "+s, prepend_bos=False)) for s in ENTITIES}
# %%
entities = valid_entities
layer = 16
batch_size = 64
thresh = 1.8
records = []
resids_list = []
for i in tqdm.trange(100):
    # Randomly choose n entries
    chosen_a = random.sample(entities, batch_size)
    chosen_b = random.sample(entities, batch_size)
    chosen_a_ext = nutils.list_flatten([[a, b] for a, b in zip(chosen_a, chosen_b)])
    chosen_b_ext = nutils.list_flatten([[b, a] for a, b in zip(chosen_a, chosen_b)])
    # print(chosen_a)
    # print(chosen_b)
    prompts = [make_prompt(a, b) for a, b in zip(chosen_a_ext, chosen_b_ext)]
    # print(prompts)

    tokens = model.to_tokens(prompts)

    logits, cache = model.run_with_cache(tokens, names_filter=utils.get_act_name("resid_post", layer))

    all_resids = cache[utils.get_act_name("resid_post", layer)]

    final_a_pos = [7+num_tokens_per_entity[a] for a in chosen_a_ext]
    final_b_pos = [11+num_tokens_per_entity[a]+num_tokens_per_entity[b] for a, b in zip(chosen_a_ext, chosen_b_ext)]
    final_pos = [19+num_tokens_per_entity[a]+num_tokens_per_entity[b] for a, b in zip(chosen_a_ext, chosen_b_ext)]
    # print(final_a_pos)
    # print(final_b_pos)
    # print(final_pos)

    resids_a = all_resids[np.arange(len(all_resids)), final_a_pos].cpu()
    resids_b = all_resids[np.arange(len(all_resids)), final_b_pos].cpu()
    # print(resids_a.shape)
    # print(resids_b.shape)

    logit_diff = logits[np.arange(len(logits)), final_pos, TF_TOKENS[0]] - logits[np.arange(len(logits)), final_pos, TF_TOKENS[1]]
    ave_diff = (logit_diff[::2] - logit_diff[1::2])/2
    consistent = (logit_diff[::2]>0)==(logit_diff[1::2]<0)

    is_useful_data = (consistent) & (ave_diff.abs()>thresh)
    label = ave_diff > 0
    is_useful_data = einops.repeat(is_useful_data, "x -> (x 2)")
    label = einops.repeat(label, "x -> (x 2)")
    label[1::2] = ~label[1::2]
    # label_kept = label[is_useful_data]
    # print(is_useful_data.float().mean())
    # print(label_kept)

    # resids_a_kept = resids_a[is_useful_data].cpu()
    # resids_b_kept = resids_b[is_useful_data].cpu()

    # print(resids_a_kept.shape)
    # print(resids_b_kept.shape)
    for i in range(len(chosen_a_ext)):
        if is_useful_data[i]:
            records.append(dict(
                a=chosen_a_ext[i],
                b=chosen_b_ext[i],
                label=label[i].item(),
                logit_diff=logit_diff[i].item(),
            ))
            resids_list.append(torch.stack([resids_a[i], resids_b[i]]))
# %%
valid_resids = torch.stack(resids_list)
valid_data_df = pd.DataFrame(records)
new_df = valid_data_df.drop_duplicates()
valid_resids_deduped = valid_resids[new_df.index]
new_df = new_df.reset_index(drop=True)
valid_resids = valid_resids_deduped
valid_data_df = new_df
torch.save(valid_resids, "valid_resids.pt")
valid_data_df.to_csv("valid_data.csv")
# %%
new_df
# %%
train_data_df = new_df
train_data_df.to_csv("train_data_dedup.csv")
train_resids = train_resids_deduped
torch.save(train_resids, "train_resids_dedup.pt")
# %%
true_ave = train_resids[train_data_df.query("label").index].float().mean(0)
true_ave = true_ave[0] - true_ave[1]
false_ave = train_resids[train_data_df.query("~label").index].float().mean(0)
false_ave = false_ave[0] - false_ave[1]
print(true_ave.shape, false_ave.shape)
# %%
nutils.cos(true_ave.float(), false_ave.float())
# %%
diff_dir = true_ave - false_ave
diff_resids = train_resids[:, 0, :] - train_resids[:, 1, :]
train_data_df["mean_diff_proj"] = to_numpy(diff_resids.float() @ diff_dir.float())
train_data_df["mean_diff_cos"] = to_numpy((diff_resids.float() @ diff_dir.float()) / diff_dir.float().norm() / diff_resids.float().norm(dim=-1))
train_data_df.sort_values("mean_diff_proj")
# %%
px.histogram(train_data_df, x="mean_diff_proj", color="label", barmode="overlay", hover_name="a", marginal="box").show()
px.histogram(train_data_df, x="mean_diff_cos", color="label", barmode="overlay", hover_name="a", marginal="box").show()
# %%

diff_dir = true_ave - false_ave
diff_resids = train_resids[:, 0, :] - train_resids[:, 1, :]
train_data_df["mean_diff_proj_a"] = to_numpy(train_resids[:, 0, :].float() @ diff_dir.float())
train_data_df["mean_diff_proj_b"] = to_numpy(train_resids[:, 1, :].float() @ diff_dir.float())
# train_data_df["mean_diff_cos"] = to_numpy((diff_resids.float() @ diff_dir.float()) / diff_dir.float().norm() / diff_resids.float().norm(dim=-1))
train_data_df.sort_values("mean_diff_proj_a")
# %%
nutils.show_df(train_data_df[["b", "mean_diff_proj_b"]].groupby("b").median().sort_values("mean_diff_proj_b").iloc[::10])
# %%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
resids_train, resids_test, labels_train, labels_test = train_test_split(to_numpy(train_resids[:, 0, :] - train_resids[:, 1, :]), train_data_df.label.to_numpy(), test_size=0.2, random_state=42)
# %%
probe = LogisticRegression(max_iter=50, C=0.1)
probe.fit(resids_train, labels_train)
labels_test_pred = probe.predict(resids_test)
(labels_test_pred==labels_test).astype(float).mean()
# %%
px.histogram(resids_test @ probe.coef_[0], color=labels_test)
# %%
train_data_df["probe_proj_a"] = to_numpy(train_resids[:, 0, :].float() @ torch.tensor(probe.coef_[0]).float())
train_data_df["probe_proj_b"] = to_numpy(train_resids[:, 1, :].float() @ torch.tensor(probe.coef_[0]).float())
# train_data_df["probe_cos"] = to_numpy((diff_resids.float() @ diff_dir.float()) / diff_dir.float().norm() / diff_resids.float().norm(dim=-1))
train_data_df.sort_values("probe_proj_a")
nutils.show_df(train_data_df[["b", "probe_proj_b"]].groupby("b").median().sort_values("probe_proj_b").iloc[::10])
# %%
labels_test_pred = probe.predict(to_numpy((valid_resids[:, 0, :] - valid_resids[:, 1, :]).float()))
(labels_test_pred==valid_data_df.label.to_numpy()).astype(float).mean()
# %%
valid_data_df["probe_proj_a"] = to_numpy(valid_resids[:, 0, :].float() @ torch.tensor(probe.coef_[0]).float())
valid_data_df["probe_proj_b"] = to_numpy(valid_resids[:, 1, :].float() @ torch.tensor(probe.coef_[0]).float())
# valid_data_df["probe_cos"] = to_numpy((diff_resids.float() @ diff_dir.float()) / diff_dir.float().norm() / diff_resids.float().norm(dim=-1))
valid_data_df.sort_values("probe_proj_a")
nutils.show_df(valid_data_df[["b", "probe_proj_b"]].groupby("b").median().sort_values("probe_proj_b").iloc[::10])

# %%
