import json
import re

from langflow.custom import Component
from langflow.io import Output, MessageTextInput
from langflow.schema import Data


class TextToObject(Component):
    display_name = "Text to Object (JSON Parse, Error Branch)"
    description = (
        "รับข้อความ Text ที่เป็น JSON string แล้ว parse เป็น Object "
        "รองรับทั้ง JSON ปกติ และ Markdown Code Block (```json)"
    )
    icon = "braces"

    inputs = [
        MessageTextInput(
            name="input_text",
            display_name="Input Text",
            info='ข้อความ JSON เช่น {"key":"value"} หรือ ```json ... ```',
            input_types=["Message"],
        )
    ]

    outputs = [
        Output(
            name="success_output",
            display_name="Success",
            method="on_success",
            group_outputs=True,
        ),
        Output(
            name="error_output",
            display_name="Error",
            method="on_error",
            group_outputs=True,
        ),
    ]

    def _clean_json_text(self, text: str) -> str:
        """ลบ Markdown Code Block ถ้ามี"""

        if text is None:
            return ""

        text = text.strip()

        # รองรับ
        # ```json
        # {...}
        # ```
        #
        # และ
        # ```
        # {...}
        # ```
        pattern = r"^```(?:json)?\s*(.*?)\s*```$"
        match = re.match(pattern, text, flags=re.DOTALL | re.IGNORECASE)

        if match:
            text = match.group(1).strip()

        return text

    def _try_parse(self):
        """พยายาม parse JSON ครั้งเดียว และ cache ผลลัพธ์"""

        cache_key = f"{self._id}_parse_result"

        if self.ctx.get(cache_key) is not None:
            return self.ctx[cache_key]

        cleaned_text = self._clean_json_text(self.input_text)

        try:
            parsed = json.loads(cleaned_text)

            result = (
                True,
                Data(
                    data=parsed
                ),
            )

        except json.JSONDecodeError as e:
            result = (
                False,
                Data(
                    data={
                        "error": True,
                        "message": str(e),
                        "raw_text": self.input_text,
                    }
                ),
            )

        self.ctx[cache_key] = result
        return result

    def on_success(self) -> Data:
        is_ok, data_obj = self._try_parse()

        if not is_ok:
            self.stop("success_output")
            return None

        self.status = data_obj
        return data_obj

    def on_error(self) -> Data:
        is_ok, data_obj = self._try_parse()

        if is_ok:
            self.stop("error_output")
            return None

        self.status = data_obj
        return data_obj