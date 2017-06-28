
def forward(self, sentence, actions=None):
        """
        Does the core parsing logic.
        If you are supplied actions, you should do those.
        Make sure to return everything that needs to be returned
            * The log probabilities from every choice made
            * The dependency graph
            * The actions you did as a list

        The boiler plate at the beginning is just gross padding and stuff
        You can basically ignore it.  Just know that it initializes a valid
        ParserState object, and now you may do actions on that state by calling
        shift(), reduce_right(), reduce_left(), or get features from it in your
        feature extractor.

        Make sure that if you are not supplied actions, and you do actions your
        network predicts, that you only do valid actions.

        Also, note that symbolic constants have been defined for the different Actions in constants.py
        E.g Actions.SHIFT is 0, Actions.REDUCE_L is 1, so that the 0th element of
        the output of your action chooser is the log probability of shift, the 1st is the log probability
        of REDUCE_L, etc.
        """
        self.refresh() # clear up hidden states from last run, if need be

        padded_sent = sentence + [END_OF_INPUT_TOK]

        # Initialize the parser state
        sentence_embs = self.word_embedding_component(padded_sent)

        parser_state = ParserState(padded_sent, sentence_embs, self.combiner, null_stack_tok_embed=self.null_stack_tok_embed)
        outputs = [] # Holds the output of each action decision
        actions_done = [] # Holds all actions we have done

        dep_graph = set() # Build this up as you go

        # Make the action queue if we have it
        if actions is not None:
            action_queue = deque()
            action_queue.extend([ Actions.action_to_ix[a] for a in actions ])
            have_gold_actions = True
        else:
            have_gold_actions = False

        # STUDENT
        if have_gold_actions:
            for i in action_queue:
            	#print i
            	actions_done.append(i)
            	if i == Actions.SHIFT:
            		parser_state.shift()
            	else:
            		#print parser_state
            		dep_graph.add(parser_state._reduce(i))
        else:
            while (parser_state.done_parsing() == False):
            	feat_extractor = feat_extractors.SimpleFeatureExtractor()
            	feats = feat_extractor.get_features(parser_state)
            	act_chooser = neural_net.ActionChooserNetwork(len(feats)*len(feats[0]))
            	log_probs = act_chooser(feats)
            	outputs.append(log_probs)
            	action_taken = utils.argmax(log_probs)
            	if action_taken != Actions.SHIFT:
            		if parser_state.stack_len()<2:
            			parser_state.shift()
            			actions_done.append(Actions.SHIFT)
            		else:
            			dep_graph.add(parser_state._reduce(action_taken))
            			actions_done.append(action_taken)
            	else:
            		if parser_state.input_buffer_len() == 1:
            			dep_graph.add(parser_state._reduce(Actions.REDUCE_R))
            			actions_done.append(Actions.REDUCE_R)
            		else:
            			parser_state.shift()
            			actions_done.append(Actions.SHIFT)
        # END STUDENT
        #print "yo"
        dep_graph.add(DepGraphEdge((ROOT_TOK, -1), (parser_state.stack[-1].headword, parser_state.stack[-1].headword_pos)))
        #print outputs, dep_graph, actions_done
        return outputs, dep_graph, actions_done
        #return "as"
        #return actions_done