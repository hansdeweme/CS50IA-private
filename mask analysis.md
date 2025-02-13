Analysis

Layer 1, Head 2

Observations:

This head pays strong attention to the [MASK] token, showing a clear focus on masked word prediction.

High attention scores are also observed for the CLS token, which might indicate an attempt to maintain global sentence meaning.

Other words have relatively distributed attention scores.

Example Sentences:

"I threw a small rock and it fell in the [MASK]."

"I was walking down the [MASK] when I saw a dog."

"My [MASK] walked in the living room while talking on the phone."

Interpretation:

This attention head may be specialized in sentence structure and masked word completion.

High CLS-token interaction suggests that context-based understanding is essential.

Layer 12, Head 9

Observations:

Displays a diagonal attention pattern, meaning words are primarily attending to themselves.

Increased attention near sentence boundaries (e.g., periods and commas).

When [MASK] is placed at different positions, more attention is directed toward punctuation.

Example Sentences:

"I threw a small rock and it fell in the [MASK]."

"I was walking down the [MASK] when I saw a dog."

"My [MASK] walked in the living room while talking on the phone."

Interpretation:

The self-referential nature suggests this head is involved in token identity preservation.

High punctuation attention indicates a boundary-aware processing function.

Additional Analysis Recommendations

Layer 5, Head 4: Potential focus on verb-object relationships.

Layer 7, Head 6: Likely attention to determiners and adjectives modifying nouns.

Future Work: Investigate how attention shifts when replacing [MASK] with ambiguous words (e.g., "bank" as in "river bank" vs. "financial bank").

