from .optimizer_base import BaseOptimizer
from ..api.palm2 import generate_text as palm_generate_text
from loguru import logger

class PalmOptimizer(BaseOptimizer):
    def fetch_new_responses(self, query_prompt):
        completion = palm_generate_text(
            model='models/text-bison-001',
            prompt=query_prompt,
            max_output_tokens=512,
            candidate_count=self.args.num_sample_per_iter,
            temperature=1.0,
        )
        candidates = completion.candidates
        candidates = [item['output'] for item in candidates]
        candidates = [item[item.find('<RES>')+5:item.find('</RES>')].strip() for item in candidates]
        candidates = [item for item in candidates if len(item) > 1]
        if len(candidates) == 0:
            logger.info(completion)
            logger.info(query_prompt)
        return candidates

    def get_name(self):
        if self.negative_optimization_goal:
            return f'optimizer-palm2-negative-{self.args.num_iter}-{self.args.num_sample_per_iter}-{self.args.num_best_sample}-{self.args.early_stop_threshold}'
        else:
            return f'optimizer-palm2-{self.args.num_iter}-{self.args.num_sample_per_iter}-{self.args.num_best_sample}-{self.args.early_stop_threshold}'


class PalmOptimizerDialog(PalmOptimizer):
    def __init__(self, args, task, gold_evaluator, victim_evaluator):
        super().__init__(args, task, gold_evaluator, victim_evaluator)
        self.meta_prompt = """
Your task is to generate a dialogue utterance <RES> as the response to the given dialogue context. Below are several previous responses with their scores, where higher scores indicate better quality.

{texts_and_scores}

The dialogue utterance <RES> can be combined with the dialogue context to form a coherent dialogue:
{context_texts}

Generate a dialogue response that is different from all the responses <RES> above, and has a higher score than all the responses <RES> above. The response should begin with <RES> and end with </RES>
"""

    def get_query_prompt(self, context, response_text_score, context2=None):
        last_utterance = context.strip().split('\n')[-1]
        if last_utterance.startswith('A: '):
            context_with_res = f'{context.strip()}\nB: <RES>'
        elif last_utterance.startswith('B: '):
            context_with_res = f'{context.strip()}\nA: <RES>'
        else:
            raise NotImplementedError
        query_prompt = self.meta_prompt.replace('{context_texts}', context_with_res)
        query_prompt = query_prompt.replace('{texts_and_scores}', response_text_score)
        return query_prompt


class PalmOptimizerSum(PalmOptimizer):
    def __init__(self, args, task, gold_evaluator, victim_evaluator):
        super().__init__(args, task, gold_evaluator, victim_evaluator)

        self.meta_prompt = """
Your task is to generate a three-sentence factually consistent summary <RES> within 40 words for the given article context.

Below are several previous summaries with their scores. The score ranges from 0 to 100, where higher scores indicate better quality.

{texts_and_scores}

The article to be summarized is as follows:
{context_texts}

Read the article carefully and generate a summary that is different from all the summaries <RES> above, and has a higher score than all the summaries <RES> above. The summary should begin with <RES> and end with </RES>. Remember to be creative and generate summary as diverse as possible.
""".strip()

    def get_query_prompt(self, context, response_text_score, context2=None):
        query_prompt = self.meta_prompt.replace('{context_texts}', context.strip())
        query_prompt = query_prompt.replace('{texts_and_scores}', response_text_score)
        return query_prompt


class PalmOptimizerQA(PalmOptimizer):
    def __init__(self, args, task, gold_evaluator, victim_evaluator):
        super().__init__(args, task, gold_evaluator, victim_evaluator)

        self.meta_prompt = """
Your task is to generate a question <RES> for the given article context and answer.
Ensure that the question can be answered using the information from the article, and that it aligns with the provided answer.

Below are several previous questions with their scores, where higher scores indicate better quality.

{texts_and_scores}

The final article, question, answer pair is as follows:
{context_texts}

Read the article carefully and generate a question that is different from all the questions <RES> above, and has a higher score than all the questions <RES> above. The question should begin with <RES> and end with </RES>. Remember to be creative and generate question as diverse as possible.
""".strip()

    def get_query_prompt(self, context, response_text_score, context2=None):
        context_with_res = f'Article:\n{context.strip()}\nQuestion:\n<RES>\nAnswer:\n{context2}'
        query_prompt = self.meta_prompt.replace('{context_texts}', context_with_res)
        query_prompt = query_prompt.replace('{texts_and_scores}', response_text_score)
        return query_prompt


class PalmOptimizerDialogSum(PalmOptimizer):
    def __init__(self, args, task, gold_evaluator, victim_evaluator):
        super().__init__(args, task, gold_evaluator, victim_evaluator)
        self.meta_prompt = """
Your task is to generate a two-sentence summary <RES> within 30 words for the given dialogue context.

Below are several previous summaries with their scores, where higher scores indicate better quality.

{texts_and_scores}

The dialogue to be summarized is as follows:
{context_texts}

Read the dialogue carefully and generate a summary that is different from all the summaries <RES> above, and has a higher score than all the summaries <RES> above. The summary should begin with <RES> and end with </RES>. Remember to be creative and generate summary as diverse as possible.
""".strip()

    def get_query_prompt(self, context, response_text_score, context2=None):
        query_prompt = self.meta_prompt.replace('{context_texts}', context.strip())
        query_prompt = query_prompt.replace('{texts_and_scores}', response_text_score)
        return query_prompt
