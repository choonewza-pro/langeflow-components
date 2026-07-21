from langflow.custom import Component
from langflow.io import Output, DataInput
from langflow.schema import Data


class NoneGuard(Component):
    display_name = "None Guard"
    description = (
        "กัน error 'Input data cannot be None' ที่เกิดจาก JsonFieldRouter branch ที่ถูก stop "
        "แต่ยังมี None หลุดมาถึง ChatOutput — ถ้าเจอ None จะไม่ส่งอะไรต่อ (stop เส้นทางนี้แทน)"
    )
    icon = "shield"

    inputs = [
        DataInput(
            name="input_data",
            display_name="Input Object",
            info="ลากสายจาก output ของ JsonFieldRouter (true_output หรือ false_output) มาต่อที่นี่ก่อนเข้า ChatOutput",
            required=False,
        ),
    ]

    outputs = [
        Output(name="passthrough", display_name="Passthrough", method="guard"),
    ]

    def guard(self) -> Data:
        if self.input_data is None:
            # เส้นทางนี้ไม่ควรทำงาน (branch ที่ถูก stop จริงๆ) -> หยุดทันที ไม่ส่งต่อให้ ChatOutput
            self.stop("passthrough")
            return None
        self.status = self.input_data
        return self.input_data
