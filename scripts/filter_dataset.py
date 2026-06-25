import polars as pl
from transformers import AutoTokenizer

keywords = [
    "sugar",
    "flour",
    "butter",
    "eggs",
    "milk",
    "onions",
    "garlic",
    "oil",
    "beef",
    "chicken",
    "gelato",
    "ice cream",
]
pattern = "|".join(keywords)

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
MAX_TOKENS = 128


def encode_and_pad(series: pl.Series) -> pl.Series:
    texts = series.to_list()
    tokenized = tokenizer(texts, max_length=MAX_TOKENS, truncation=True, add_special_tokens=True)["input_ids"]
    return pl.Series([" ".join(map(str, tokens)) for tokens in tokenized], dtype=pl.String)


def check_token_length(series: pl.Series) -> pl.Series:
    texts = series.to_list()
    tokenized = tokenizer(texts, truncation=False, add_special_tokens=True)["input_ids"]
    return pl.Series([len(tokens) <= MAX_TOKENS for tokens in tokenized], dtype=pl.Boolean)


def extract_action_context(series: pl.Series) -> pl.Series:
    return series.str.split(" ").list.slice(0, 4).list.join(" ")


print("Streaming and building pre-tokenized target blocks...")
lazy_data = pl.scan_csv("data/RecipeNLG_dataset.csv")

data_sampled = (
    lazy_data
    .filter(pl.col("ingredients").str.contains(pattern))
    .unique()
    .limit(100000)
    .filter(
        pl.col("ingredients").map_batches(check_token_length) & pl.col("directions").map_batches(check_token_length)
    )
    .collect()
    .sample(fraction=1.0, shuffle=True, seed=42)
    .limit(20000)
    .with_columns([
        # Generate an action proxy from the first few words of the directions
        pl.col("directions").map_batches(extract_action_context).alias("action_text")
    ])
)

# ==============================================================================
# NEW: Synthetic State-Action-State Triples Block
# ==============================================================================
# Define your base food components and how major actions transition them.
triples_raw = [
    # --- COOK (General Heat / Simmer / Steam) ---
    {"ingredients": "pasta water", "action_text": "cook", "directions": "cooked al dente pasta"},
    {"ingredients": "rice water", "action_text": "cook", "directions": "fluffy cooked rice"},
    {"ingredients": "vegetables", "action_text": "cook", "directions": "steamed tender vegetables"},
    {"ingredients": "oatmeal milk", "action_text": "cook", "directions": "warm prepared porridge"},
    {"ingredients": "quinoa water", "action_text": "cook", "directions": "cooked quinoa grains"},
    # --- FRY (Pan-fry, Deep-fry, Sauté, Stir-fry) ---
    {"ingredients": "chicken breast oil", "action_text": "fry", "directions": "golden pan fried chicken"},
    {"ingredients": "potatoes oil salt", "action_text": "fry", "directions": "crispy potato french fries"},
    {"ingredients": "onions butter", "action_text": "fry", "directions": "sautéed caramelized onions"},
    {"ingredients": "beef strips garlic oil", "action_text": "fry", "directions": "savory stir fried beef"},
    {"ingredients": "eggs butter", "action_text": "fry", "directions": "fried sunny side up eggs"},
    {"ingredients": "bacon strips", "action_text": "fry", "directions": "crispy rendered bacon"},
    {"ingredients": "mushrooms oil", "action_text": "fry", "directions": "sautéed brown mushrooms"},
    {"ingredients": "tofu cubes oil", "action_text": "fry", "directions": "crispy deep fried tofu"},
    # --- BAKE (Oven Dry Heat / Rising) ---
    {"ingredients": "dough flour sugar butter eggs", "action_text": "bake", "directions": "baked cake cookie bread"},
    {"ingredients": "potatoes cheese cream", "action_text": "bake", "directions": "baked potato gratin casserole"},
    {"ingredients": "apples cinnamon sugar", "action_text": "bake", "directions": "baked apple pie dessert"},
    {"ingredients": "fish fillet lemon herbs", "action_text": "bake", "directions": "oven baked tender fish"},
    {"ingredients": "macaroni cheese milk", "action_text": "bake", "directions": "baked macaroni and cheese"},
    {"ingredients": "batter flour cocoa eggs", "action_text": "bake", "directions": "fudge chocolate brownies"},
    # --- BOIL (High Temp Liquid) ---
    {"ingredients": "raw eggs water", "action_text": "boil", "directions": "hard boiled eggs"},
    {"ingredients": "whole potatoes water", "action_text": "boil", "directions": "soft boiled potatoes"},
    {"ingredients": "shrimp water salt", "action_text": "boil", "directions": "pink cooked tender shrimp"},
    {"ingredients": "corn cob water", "action_text": "boil", "directions": "tender boiled sweet corn"},
    # --- GRILL (Direct Open Heat) ---
    {"ingredients": "beef patty", "action_text": "grill", "directions": "charbroiled grilled hamburger burger"},
    {"ingredients": "steak oil pepper", "action_text": "grill", "directions": "seared medium rare grilled steak"},
    {
        "ingredients": "chicken wings barbecue sauce",
        "action_text": "grill",
        "directions": "smoky barbecued grilled wings",
    },
    {"ingredients": "zucchini peppers asparagus", "action_text": "grill", "directions": "charred grilled vegetables"},
    {"ingredients": "salmon fillet oil", "action_text": "grill", "directions": "smoky grilled salmon skin"},
    # --- ROAST (High Heat Oven Browning) ---
    {"ingredients": "whole chicken herbs", "action_text": "roast", "directions": "crispy roasted whole chicken"},
    {
        "ingredients": "carrots potatoes oil rosemary",
        "action_text": "roast",
        "directions": "oven roasted root vegetables",
    },
    {"ingredients": "pork loin garlic", "action_text": "roast", "directions": "tender roasted pork tenderloin"},
    {"ingredients": "nuts seeds", "action_text": "roast", "directions": "toasty roasted nuts"},
    # --- CHOP / CUT (Mechanical Alteration) ---
    {"ingredients": "whole onions", "action_text": "chop", "directions": "finely diced chopped onions"},
    {"ingredients": "tomatoes", "action_text": "chop", "directions": "diced tomato chunks"},
    {"ingredients": "carrots", "action_text": "chop", "directions": "sliced carrot coins"},
    {"ingredients": "parsley herbs", "action_text": "chop", "directions": "minced fresh green herbs"},
    {"ingredients": "bread loaf", "action_text": "chop", "directions": "sliced pieces of bread"},
    {"ingredients": "cheese block", "action_text": "chop", "directions": "shredded grated cheese"},
    # --- MIX / BLEND (Homogenization) ---
    {"ingredients": "flour milk eggs", "action_text": "mix", "directions": "smooth liquid pancake batter"},
    {"ingredients": "oil vinegar mustard", "action_text": "mix", "directions": "emulsified salad dressing vinaigrette"},
    {"ingredients": "lettuce tomatoes cucumbers", "action_text": "mix", "directions": "tossed mixed green salad"},
    {"ingredients": "strawberries milk yogurt ice", "action_text": "mix", "directions": "blended fruit smoothie"},
    {"ingredients": "cream sugar vanilla", "action_text": "mix", "directions": "whipped heavy cream paste"},
    # --- MELT (Phase Change - Solid to Liquid) ---
    {"ingredients": "butter stick", "action_text": "melt", "directions": "liquid melted yellow butter"},
    {"ingredients": "chocolate bar", "action_text": "melt", "directions": "smooth glossy molten chocolate"},
    {"ingredients": "cheese slices", "action_text": "melt", "directions": "gooey melted warm cheese"},
    # --- FREEZE (Phase Change - Liquid to Solid) ---
    {"ingredients": "water", "action_text": "freeze", "directions": "solid frozen ice cubes"},
    {"ingredients": "fruit juice sugar", "action_text": "freeze", "directions": "frozen sweet popsicles ice"},
    {"ingredients": "cream milk sugar churned", "action_text": "freeze", "directions": "cold frozen ice cream scoop"},
    # --- TOAST (Surface Browning) ---
    {"ingredients": "bread slice", "action_text": "toast", "directions": "crispy brown toasted bread"},
    {"ingredients": "marshmallow", "action_text": "toast", "directions": "gooey roasted toasted marshmallow"},
]

# Create an identical schema DataFrame for the synthetic triples
synthetic_df = pl.DataFrame(
    triples_raw, schema={"ingredients": pl.String, "action_text": pl.String, "directions": pl.String}
)

# Use select matching to ensure column structural ordering matches perfectly
data_sampled = data_sampled.select(["ingredients", "action_text", "directions"])
synthetic_df = synthetic_df.select(["ingredients", "action_text", "directions"])

# Append the synthetic foundational world-knowledge to your real dataset
data_sampled = data_sampled.vstack(synthetic_df)
data_sampled.write_csv("data/recipe_sampled.csv")
# ==============================================================================

# Apply tokenization transformations across all 3 components (including synthetic)
data_sampled = data_sampled.with_columns([
    pl.col("ingredients").map_batches(encode_and_pad).alias("x_tokens"),
    pl.col("action_text").map_batches(encode_and_pad).alias("a_tokens"),
    pl.col("directions").map_batches(encode_and_pad).alias("y_tokens"),
])

data_sampled.select(["x_tokens", "a_tokens", "y_tokens"]).write_parquet("data/recipe_sampled.parquet")
print(f"Done! Saved {len(data_sampled)} records (including {len(synthetic_df)} explicit action triples).")
