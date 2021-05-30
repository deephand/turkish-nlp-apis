from tensorflow.keras.models import load_model
import numpy as np

class Model:
    def __init__(self):
        folder = '/'.join(__file__.split('/')[:-1])
        self.decoder = load_model(folder + '/' + 'decoder.h5')
        self.encoder = load_model(folder + '/' + 'encoder.h5')

        self.characters = ['END'] + list('abcdefghijklmnopqrstuvwxyzöçşığü')
        self.reverse_token_map = dict([(i , char) for i, char in enumerate(self.characters)])
        self.token_index_map = dict([(self.reverse_token_map[key], key) for key in self.reverse_token_map])
    
    def predict_root_word(self, word):
        def convert_word_to_np_array(word):
            word_input_encoding = np.zeros((1, len(word), len(self.token_index_map)))
            for char_idx, char in enumerate(word):
                word_input_encoding[0, char_idx, self.token_index_map[char]] = 1

            return word_input_encoding
        
        word_np = convert_word_to_np_array(word)
        encoder_output = self.encoder.predict(word_np)

        decoder_start = np.zeros((1, 1, len(self.token_index_map)))

        best_char_index = -1
        decoder_input = [decoder_start] + encoder_output
        
        predict_word = ''
        while best_char_index != self.token_index_map['END']:
            decoder_output = self.decoder.predict(decoder_input)
            char_probs = decoder_output[0]

            best_char_index = np.argmax(char_probs[0, 0, :])
            predict_word += self.reverse_token_map[best_char_index]
            
            char_probs[0, 0, :] = 0.0
            char_probs[0, 0, best_char_index] = 1.0 
            decoder_input = [char_probs] + decoder_output[1:]
        return predict_word[:-3]
    
    def predict(self, words):
        words_list = words.split()
        words_list = [self.filter_word(word) for word in words_list]
        return ' '.join([self.predict_root_word(word) for word in words_list])
    
    def filter_word(self, word):
        return ''.join([chr for chr in word if chr in self.token_index_map])
