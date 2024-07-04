from .bertscore import BERTSCOREEvaluator
from .sacrebleu import SACREBLEUEvaluator
from .poe import PoEEvaluator
from .bleurt import BLEURTEvaluator
from .llmeval import LLMDialogEvaluator, LLMDialogSumEvaluator, LLMSumEvaluator, LLMQAEvaluator
from .unieval import DialogEvaluator as UniDialogEvaluator
from .unieval import SumEvaluator as UniSumEvaluator
from .bartscore import BARTSCOREEvaluator
from .rouge import ROUGEEvaluator
from .rquge import RQUGEEvaluator
from .true import TrueEvaluator
from .chatgpt import (
    ChatGPTDialogEvaluator,
    ChatGPTQAEvaluator,
    ChatGPTSumEvaluator,
    ChatGPTDialogSumEvaluator
)
from .palm import (
    PalmDialogEvaluator,
    PalmQAEvaluator,
    PalmDialogSumEvaluator,
    PalmSumEvaluator,
)

evaluator_list = {
    'rouge': ROUGEEvaluator,
    'rquge': RQUGEEvaluator,
    'true': TrueEvaluator,
    'palm_dialog': PalmDialogEvaluator,
    'palm_qa': PalmQAEvaluator,
    'palm_sum': PalmSumEvaluator,
    'palm_dialogsum': PalmDialogSumEvaluator,
    'bertscore': BERTSCOREEvaluator,
    'bartscore': BARTSCOREEvaluator,
    'sacrebleu': SACREBLEUEvaluator,
    'bleurt': BLEURTEvaluator,
    'poe': PoEEvaluator,
    'chatgpt_dialog': ChatGPTDialogEvaluator,
    'chatgpt_qa': ChatGPTQAEvaluator,
    'chatgpt_sum': ChatGPTSumEvaluator,
    'chatgpt_dialogsum': ChatGPTDialogSumEvaluator,
    'llm_dialog': LLMDialogEvaluator,
    'llm_dialogsum': LLMDialogSumEvaluator,
    'llm_sum': LLMSumEvaluator,
    'llm_qa': LLMQAEvaluator,
    'unieval_dialog': UniDialogEvaluator,
    'unieval_sum': UniSumEvaluator,
}
