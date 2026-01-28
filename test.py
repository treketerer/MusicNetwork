from miditok import REMI, TokenizerConfig
from symusic import Score

# Creating a multitrack tokenizer, read the doc to explore all the parameters
config = TokenizerConfig(num_velocities=16, use_chords=True, use_programs=True)
tokenizer = REMI(config)

# Loads a midi, converts to tokens, and back to a MIDI
midi = Score("midi/00000b8982c198eb5ba5540ac0aa6358.mid")
tokens = tokenizer(midi)  # calling the tokenizer will automatically detect MIDIs, paths and tokens
print(tokenizer.vocab)
# converted_back_midi = tokenizer(tokens)
