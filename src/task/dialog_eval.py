
class BaseTaskForResponseEvaluation:
    def __init__(self, args, evaluators):
        self.args = args

    def preprocess(self, raw_dataset):
        processed_dataset = []
        if self.args.context_len == -1:
            for dialog in raw_dataset['dialog']:
                current_context = ''
                for idx, utterance in enumerate(dialog[:-1]):
                    if self.args.no_role:
                        current_context += f'{utterance.strip()}\n'
                    else:
                        if idx % 2 == 0:
                            current_context += f'A: {utterance.strip()}\n'
                        else:
                            current_context += f'B: {utterance.strip()}\n'
                    current_response = dialog[idx + 1]

                    processed_dataset.append(
                        [current_context.strip() + ('\n\n' if self.args.no_role else ''), current_response.strip()]
                    )
        else:
            for dialog in raw_dataset['dialog']:
                if len(dialog) < self.args.context_len + 1:
                    continue
                context = ''
                for idx, utterance in enumerate(dialog[:self.args.context_len]):
                    if self.args.no_role:
                        context += f'{utterance.strip()}\n'
                    else:
                        if idx % 2 == 0:
                            context += f'A: {utterance.strip()}\n'
                        else:
                            context += f'B: {utterance.strip()}\n'
                response = dialog[self.args.context_len]
                processed_dataset.append(
                    [context.strip() + ('\n\n' if self.args.no_role else ''), response]
                )
        return processed_dataset

    @staticmethod
    def add_args(parser):
        parser.add_argument('--context_len', default=5, help='number of utterances in the context')
        # for unieval
        parser.add_argument('--no_role', action='store_true')
        return parser
