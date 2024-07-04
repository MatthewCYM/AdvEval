from .daily_dialog import DailyDialogForResponseEvaluation
from .persona_chat import PersonaChatForResponseEvaluation
from .cnn_dailymail import CNNDailyMailForSummarizationEvaluation
from .mutual import MuTualForResponseEvaluation
from .xsum import XSumForSummarizationEvaluation
from .dialogsum import DialogSumForSummarizationEvaluation
from .dream import DreamForResponseEvaluation
from .squad import SquadForQAEvaluation
from .natural_question import NaturalQuestionForQAEvaluation
from .newsqa import NewsQAForQAEvaluation
from .hotpotqa import HotpotQAForQAEvaluation

task_list = {
    'response_dailydialog': DailyDialogForResponseEvaluation,
    'response_personachat': PersonaChatForResponseEvaluation,
    'response_mutual': MuTualForResponseEvaluation,
    'response_dream': DreamForResponseEvaluation,
    'sum_cnndailymail': CNNDailyMailForSummarizationEvaluation,
    'sum_xsum': XSumForSummarizationEvaluation,
    'sum_dialogsum': DialogSumForSummarizationEvaluation,
    'qa_squad': SquadForQAEvaluation,
    'qa_naturalquestion': NaturalQuestionForQAEvaluation,
    'qa_newsqa': NewsQAForQAEvaluation,
    'qa_hotpotqa': HotpotQAForQAEvaluation,
}