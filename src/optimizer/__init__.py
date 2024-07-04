from .optimizer_chatgpt import (
    ChatGPTOptimizerDialog,
    ChatGPTOptimizerQA,
    ChatGPTOptimizerSum,
    ChatGPTOptimizerDialogSum
)
from .optimizer_palm import (
    PalmOptimizerDialog,
    PalmOptimizerQA,
    PalmOptimizerSum,
    PalmOptimizerDialogSum
)

optimizer_list = {
    'optimizer_dialog_palm': PalmOptimizerDialog,
    'optimizer_dialog_gpt4': ChatGPTOptimizerDialog,
    'optimizer_qa_palm': PalmOptimizerQA,
    'optimizer_qa_gpt4': ChatGPTOptimizerQA,
    'optimizer_sum_palm': PalmOptimizerSum,
    'optimizer_sum_gpt4': ChatGPTOptimizerSum,
    'optimizer_dialogsum_palm': PalmOptimizerDialogSum,
    'optimizer_dialogsum_gpt4': ChatGPTOptimizerDialogSum,
}

