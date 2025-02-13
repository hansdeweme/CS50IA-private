import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer, TFBertForMaskedLM

# Pre-trained masked language model
MODEL = "bert-base-uncased"

# Number of predictions to generate
K = 3

# Load tokenizer globally for efficiency
tokenizer = AutoTokenizer.from_pretrained(MODEL)

# Constants for generating attention diagrams
GRID_SIZE = 40
PIXELS_PER_WORD = 200

def main():
    text = input("Text: ")

    # Tokenize input
    inputs = tokenizer(text, return_tensors="tf")
    mask_token_index = get_mask_token_index(tokenizer.mask_token_id, inputs)
    if mask_token_index is None:
        sys.exit(f"Input must include mask token {tokenizer.mask_token}.")

    # Use model to process input
    model = TFBertForMaskedLM.from_pretrained(MODEL)
    result = model(**inputs, output_attentions=True)

    # Generate predictions
    mask_token_logits = result.logits[0, mask_token_index]
    top_tokens = tf.math.top_k(mask_token_logits, K).indices.numpy()
    for token in top_tokens:
        print(text.replace(tokenizer.mask_token, tokenizer.decode([token])))

    # Visualize attentions
    visualize_attentions(inputs.tokens(), result.attentions)

def get_mask_token_index(mask_token_id, inputs):
    """Efficiently retrieve index of [MASK] token."""
    for i, token in enumerate(inputs.input_ids[0]):
        if token == mask_token_id:
            return i
    return None

def get_color_for_attention_score(attention_score):
    """
    Return a tuple of three integers representing a shade of gray for the
    given `attention_score`. Each value should be in the range [0, 255].
    """
    attention_score = attention_score.numpy()
    grayscale_value = round(attention_score * 255)  # Restore linear scaling
    return (grayscale_value, grayscale_value, grayscale_value)

def visualize_attentions(tokens, attentions):
    """Dynamically adjust text size and generate attention diagrams for all layers and heads."""
    global PIXELS_PER_WORD
    PIXELS_PER_WORD = max(150, 400 // len(tokens))  # Scale dynamically
    
    for i, layer in enumerate(attentions):
        for k in range(len(layer[0])):  # Iterate through all attention heads
            layer_number = i + 1
            head_number = k + 1
            generate_diagram(
                layer_number,
                head_number,
                tokens,
                attentions[i][0][k]
            )

def generate_diagram(layer_number, head_number, tokens, attention_weights):
    """Generate a diagram representing the self-attention scores for a single attention head."""
    image_size = GRID_SIZE * len(tokens) + PIXELS_PER_WORD
    img = Image.new("RGBA", (image_size, image_size), "black")
    draw = ImageDraw.Draw(img)

    # Draw tokens
    for i, token in enumerate(tokens):
        token_image = Image.new("RGBA", (image_size, image_size), (0, 0, 0, 0))
        token_draw = ImageDraw.Draw(token_image)
        token_draw.text(
            (image_size - PIXELS_PER_WORD, PIXELS_PER_WORD + i * GRID_SIZE),
            token,
            fill="white"
        )
        token_image = token_image.rotate(90)
        img.paste(token_image, mask=token_image)
        draw.text(
            (PIXELS_PER_WORD - 50, PIXELS_PER_WORD + i * GRID_SIZE),
            token,
            fill="white"
        )

    # Draw attention scores
    for i in range(len(tokens)):
        y = PIXELS_PER_WORD + i * GRID_SIZE
        for j in range(len(tokens)):
            x = PIXELS_PER_WORD + j * GRID_SIZE
            color = get_color_for_attention_score(attention_weights[i][j])
            draw.rectangle((x, y, x + GRID_SIZE, y + GRID_SIZE), fill=color)

    # Save image
    img.save(f"Attention_Layer{layer_number}_Head{head_number}.png")

if __name__ == "__main__":
    main()
