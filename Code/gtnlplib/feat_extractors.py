class SimpleFeatureExtractor:

    # def __init__(self):
    #     self.name = "FE"

    def get_features(self, parser_state, **kwargs):
        """
        Take in all the information and return features.
        Your features should be autograd.Variable objects
        of embeddings.

        the **kwargs is special python syntax.  You can pass additional keyword arguments,
        and they will show up in kwargs in your function body as a dict.  For example:
        ** function call **
        get_features(my_parser_state, action_history=[SHIFT, SHIFT])
        ** in function body **
        kwargs["action_history"] == [SHIFT, SHIFT] # true

        check the python documentation if you are confused.
        YOU DONT NEED TO USE KWARGS IN THIS FUNCTION!!!
        It is there only so that if you make a more complicated feature extractor for your Kaggle submission,
        you can pass in arbitrary information about the parse without this feature extractor complaining
        
        :param parser_state the ParserState object for the current parse (giving access
            to the stack and input buffer)
        :return A list of autograd.Variable objects, which are the embeddings of your
            features
        """
        #pass
        out = [0]*3
        x = parser_state.stack_peek_n(2)
        out[0],out[1]= x[0][2],x[1][2]
        out[2] = parser_state.input_buffer_peek_n(1)[0][2]
        return out
