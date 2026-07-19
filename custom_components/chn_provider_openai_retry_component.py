import json
import time
from typing import Any

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from pydantic.v1 import SecretStr

from lfx.custom.custom_component.component import Component
from lfx.field_typing.range_spec import RangeSpec
from lfx.inputs.inputs import (
    BoolInput,
    IntInput,
    MessageInput,
    MultilineInput,
    SecretStrInput,
    SliderInput,
    StrInput,
)
from lfx.io import Output
from lfx.log.logger import logger
from lfx.schema.data import Data
from lfx.schema.message import Message


class OpenAIJSONRetryComponent(Component):
    display_name = "CHN: OpenAI (JSON + Retry)"
    description = (
        "เรียก OpenAI (LLM - OpenAI Format) ให้ตอบเป็น JSON แล้ว parse ทันที ถ้า parse ไม่สำเร็จ "
        "จะเรียกซ้ำอัตโนมัติสูงสุดตามจำนวนที่กำหนด ถ้าครบจำนวนแล้วยังไม่ผ่าน จะส่งออกทาง Error"
    )
    icon = "Repeat"
    name = "OpenAIJSONRetry"

    inputs = [
        MessageInput(
            name="input_value",
            display_name="Input",
            info="ข้อความ + รูปภาพที่ผู้ใช้ส่งมาทาง Chat Input",
            required=True,
        ),
        BoolInput(
            name="require_image",
            display_name="Require Image",
            info="ถ้าเปิดไว้ (True) ต้องมีรูปภาพแนบมากับ Input เสมอ ไม่งั้นจะ error ทันทีโดยไม่เรียก LLM "
            "ถ้าปิด (False) จะทำงานแบบข้อความล้วนได้ตามปกติ",
            value=True,
        ),
        MultilineInput(
            name="system_message",
            display_name="System Message",
            info="System prompt ที่กำหนดพฤติกรรมของ LLM",
        ),
        StrInput(
            name="model_name",
            display_name="Model Name",
            value="gpt-5-mini",
            required=True,
            info="ชื่อ model เช่น gpt-5-mini",
        ),
        StrInput(
            name="openai_api_base",
            display_name="OpenAI API Base",
            info="Base URL ของ NT MATCHA / AI Gateway",
            show=True,
        ),
        SecretStrInput(
            name="api_key",
            display_name="OpenAI API Key",
            required=True,
            show=True,
        ),
        SliderInput(
            name="temperature",
            display_name="Temperature",
            value=0.3,
            range_spec=RangeSpec(min=0, max=2, step=0.01),
        ),
        IntInput(
            name="max_tokens",
            display_name="Max Tokens",
            advanced=True,
            info="ตั้งเป็น 0 หรือเว้นว่างเพื่อไม่จำกัด",
            range_spec=RangeSpec(min=0, max=128000),
        ),
        SliderInput(
            name="top_p",
            display_name="Top P",
            value=1.0,
            range_spec=RangeSpec(min=0, max=1, step=0.01),
            advanced=True,
        ),
        SliderInput(
            name="frequency_penalty",
            display_name="Frequency Penalty",
            value=0.0,
            range_spec=RangeSpec(min=-2, max=2, step=0.01),
            advanced=True,
            show=True,
        ),
        SliderInput(
            name="presence_penalty",
            display_name="Presence Penalty",
            value=0.0,
            range_spec=RangeSpec(min=-2, max=2, step=0.01),
            advanced=True,
            show=True,
        ),
        IntInput(
            name="max_retries",
            display_name="Max Retries",
            value=10,
            info="จำนวนครั้งสูงสุดที่จะเรียกซ้ำ เมื่อเจอ timeout/error หรือ parse JSON ไม่สำเร็จ",
        ),
        IntInput(
            name="timeout",
            display_name="Request Timeout (วินาที)",
            value=600,
            info="ระยะเวลาสูงสุดต่อการเรียก 1 ครั้ง เกินเวลานี้ถือว่า timeout แล้ว retry ใหม่",
        ),
    ]

    outputs = [
        Output(display_name="Success", name="success", method="run_success", group_outputs=True),
        Output(display_name="Error", name="error", method="run_error", group_outputs=True),
    ]

    def _build_llm(self) -> Any:
        api_key_value = None
        if self.api_key:
            api_key_value = (
                self.api_key.get_secret_value() if isinstance(self.api_key, SecretStr) else str(self.api_key)
            )

        return ChatOpenAI(
            api_key=api_key_value,
            model=self.model_name,
            base_url=self.openai_api_base or "https://api.openai.com/v1",
            temperature=self.temperature if self.temperature is not None else 0.3,
            max_tokens=self.max_tokens or None,
            top_p=self.top_p if self.top_p is not None else 1.0,
            frequency_penalty=self.frequency_penalty or 0.0,
            presence_penalty=self.presence_penalty or 0.0,
            timeout=self.timeout,
            max_retries=0,  # retry คุมเองใน _call_llm_with_retry เพื่อไม่ให้ retry ซ้อนกัน 2 ชั้น
            model_kwargs={"response_format": {"type": "json_object"}},
        )

    def _has_image(self) -> bool:
        """เช็คว่า Message ที่ได้จาก Chat Input มีไฟล์รูปแนบมาไหม."""
        return bool(getattr(self.input_value, "files", None))

    def _build_messages(self) -> list[Any]:
        """สร้าง list ของ LangChain messages โดยใช้ Message.to_lc_message() ของ Langflow เอง."""
        messages: list[Any] = []
        if self.system_message:
            messages.append(SystemMessage(content=self.system_message))
        messages.append(self.input_value.to_lc_message(model_name=self.model_name))
        return messages

    def _call_llm_with_retry(self) -> str:
        """เรียก LLM พร้อม retry logic ทั้งกรณี call ล้มเหลว (เช่น timeout) และ JSON parse ไม่ผ่าน."""
        max_retries = self.max_retries or 1
        llm = self._build_llm()
        messages = self._build_messages()

        last_error: Exception | None = None
        attempts_log: list[dict[str, Any]] = []

        for attempt in range(1, max_retries + 1):
            start = time.monotonic()
            try:
                response = llm.invoke(messages)
                content = response.content if hasattr(response, "content") else str(response)

                # validate ว่า parse เป็น JSON ได้จริง แต่ output ที่ส่งออกยังเป็น text ดิบ
                json.loads(content)

                attempts_log.append(
                    {"attempt": attempt, "status": "success", "elapsed_sec": round(time.monotonic() - start, 2)}
                )
                self.ctx["nt_matcha_attempts"] = attempts_log
                self.status = f"สำเร็จในครั้งที่ {attempt}/{max_retries}"
                return content

            except json.JSONDecodeError as e:
                last_error = e
                attempts_log.append(
                    {
                        "attempt": attempt,
                        "status": "json_parse_failed",
                        "error": str(e),
                        "elapsed_sec": round(time.monotonic() - start, 2),
                    }
                )
                logger.warning(f"[NT MATCHA] ครั้งที่ {attempt}/{max_retries}: parse JSON ไม่สำเร็จ - {e}")

            except Exception as e:  # noqa: BLE001
                last_error = e
                attempts_log.append(
                    {
                        "attempt": attempt,
                        "status": "call_failed",
                        "error": str(e),
                        "elapsed_sec": round(time.monotonic() - start, 2),
                    }
                )
                logger.warning(f"[NT MATCHA] ครั้งที่ {attempt}/{max_retries}: เรียก LLM ไม่สำเร็จ - {e}")

        self.ctx["nt_matcha_attempts"] = attempts_log
        error_message = f"เรียก NT MATCHA ไม่สำเร็จหลังจาก retry ครบ {max_retries} ครั้ง. Error ล่าสุด: {last_error}"
        raise RuntimeError(error_message)

    def _get_result(self) -> tuple[str | None, str | None]:
        """คืนค่า (content, error_message) — สำเร็จ error_message จะเป็น None."""
        if "nt_matcha_result" in self.ctx:
            return self.ctx["nt_matcha_result"]

        if self.require_image and not self._has_image():
            error_message = "ไม่พบรูปภาพที่ผู้ใช้ส่งมาทาง Chat Input กรุณาแนบรูปภาพก่อนส่งข้อความ"
            self.ctx["nt_matcha_attempts"] = []
            result: tuple[str | None, str | None] = (None, error_message)
            self.ctx["nt_matcha_result"] = result
            self.status = error_message
            return result

        try:
            content = self._call_llm_with_retry()
            result = (content, None)
        except Exception as e:  # noqa: BLE001
            result = (None, str(e))

        self.ctx["nt_matcha_result"] = result
        return result

    def run_success(self) -> Message:
        content, error = self._get_result()
        if error is not None:
            self.status = f"ล้มเหลว: {error}"
            self.stop("success")
            return Message(text="")

        message = Message(text=content)
        self.status = message
        return message

    def run_error(self) -> Data:
        content, error = self._get_result()
        if error is None:
            self.stop("error")
            return Data(data={})

        error_data = Data(
            data={
                "error": error,
                "error_type": "missing_image" if "ไม่พบรูปภาพ" in error else "llm_call_failed",
                "model_name": self.model_name,
                "max_retries": self.max_retries,
                "attempts": self.ctx.get("nt_matcha_attempts", []),
            }
        )
        self.status = error_data
        return error_data