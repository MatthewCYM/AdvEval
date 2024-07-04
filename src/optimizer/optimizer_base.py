from loguru import logger


class BaseOptimizer:
    def __init__(
        self,
        args,
        task,
        gold_evaluator,
        victim_evaluator
    ):
        self.args = args
        self.task = task
        self.gold_evaluator = gold_evaluator
        self.victim_evaluator = victim_evaluator
        self.negative_optimization_goal = args.negative_optimization_goal
        if self.negative_optimization_goal:
            logger.info('use negative optimization goal')
        self.meta_prompt = None

    def get_name(self):
        raise NotImplementedError

    @staticmethod
    def add_args(parser):
        parser.add_argument('--num_iter', type=int, default=30)
        parser.add_argument('--num_sample_per_iter', type=int, default=8)
        parser.add_argument('--num_best_sample', type=int, default=10)
        parser.add_argument('--early_stop_threshold', type=float, default=80)
        parser.add_argument('--negative_optimization_goal', action='store_true')

    def combine_score(self, gold_score, victim_score):
        if self.negative_optimization_goal:
            return round(victim_score - gold_score, 1)
        else:
            return round(gold_score - victim_score, 1)

    def evaluate(self):
        dataset = self.task.dataset
        rtn = {
            'context': [],
            'init_response': [],
            'adv_response': [],
            'init_gold_score': [],
            'adv_gold_score': [],
            'init_victim_score': [],
            'adv_victim_score': [],
        }

        for item in dataset:
            if len(item) == 2:
                context, init_response = item
                context, init_response, ground_truth = context.strip(), init_response.strip(), init_response.strip()
                context2 = None
            elif len(item) == 3:
                if 'context2' not in rtn:
                    rtn['context2'] = []
                # for qa tasks
                context, init_response, context2 = item
                context, init_response, ground_truth, context2 = context.strip(), init_response.strip(), init_response.strip(), context2.strip()
            else:
                raise NotImplementedError
            perturbed_response = self.perturb(context, init_response, ground_truth, context2)

            init_gold_score = self.gold_evaluator.get_score(context, init_response, ground_truth, context2)
            adv_gold_score = self.gold_evaluator.get_score(context, perturbed_response, ground_truth, context2)
            init_victim_score = self.victim_evaluator.get_score(context, init_response, ground_truth, context2)
            adv_victim_score = self.victim_evaluator.get_score(context, perturbed_response, ground_truth, context2)
            logger.info(
                f'gold score: {init_gold_score} -> {adv_gold_score}'
            )
            logger.info(
                f'victim score: {init_victim_score} -> {adv_victim_score}'
            )
            logger.info('********************************************')

            rtn['context'].append(context)
            if context2 is not None:
                rtn['context2'].append(context2)
            rtn['init_response'].append(init_response)
            rtn['adv_response'].append(perturbed_response)
            rtn['init_gold_score'].append(init_gold_score)
            rtn['adv_gold_score'].append(adv_gold_score)
            rtn['init_victim_score'].append(init_victim_score)
            rtn['adv_victim_score'].append(adv_victim_score)

        return rtn

    def get_query_prompt(self, context, response_text_score, context2=None):
        raise NotImplementedError

    def fetch_new_responses(self, query_prompt):
        raise NotImplementedError

    def perturb(self, context, init_response, ground_truth, context2=None):
        gold_score = self.gold_evaluator.get_score(context, init_response, ground_truth, context2)
        victim_score = self.victim_evaluator.get_score(context, init_response, ground_truth, context2)
        response_list = [[init_response, self.combine_score(gold_score, victim_score)]]
        response_set = set()
        response_set.add(init_response)

        for _ in range(self.args.num_iter):
            response_list = sorted(response_list, key=lambda x: x[1], reverse=False)

            # > 90 then early stop
            if response_list[-1][1] > self.args.early_stop_threshold:
                logger.info(f'final query samples={len(response_list)}')
                return response_list[-1][0]

            response_text_score = ''
            if self.args.num_best_sample != -1:
                if len(response_list) > self.args.num_best_sample:
                    current_response_list = response_list[-self.args.num_best_sample:]
                else:
                    current_response_list = response_list
            else:
                current_response_list = response_list

            for item in current_response_list:
                response, score = item[0], item[1]
                response_text_score += f'text:\n{response}\nscore:\n{score}\n'

            response_text_score = response_text_score.strip()

            query_prompt = self.get_query_prompt(context, response_text_score, context2)

            new_responses = self.fetch_new_responses(query_prompt)
            if len(new_responses) == 0:
                break

            for new_response in new_responses:
                # deduplicate
                if new_response in response_set:
                    continue
                else:
                    response_set.add(new_response)

                gold_score = self.gold_evaluator.get_score(context, new_response, ground_truth, context2)
                victim_score = self.victim_evaluator.get_score(context, new_response, ground_truth, context2)

                # gold score = -1 refers to unsafe response
                if gold_score == -1:
                    continue
                if victim_score == -1:
                    continue
                response_list.append([new_response, self.combine_score(gold_score, victim_score)])
        response_list = sorted(response_list, key=lambda x: x[1], reverse=False)
        logger.info(f'final query samples={len(response_list)}')
        best_response = response_list[-1][0]
        return best_response
