from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

LOGGER = logging.getLogger(__name__)


class GeminiService:
    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or LOGGER
        self.api_key = os.getenv('GEMINI_API_KEY')
        self.client = None
        self.model_name = 'gemini-2.5-flash'
        self.generative_model = None
        self.disabled = False

        if not self.api_key:
            self.logger.warning('GEMINI_API_KEY is not set. Gemini service will be unavailable.')
            return

        try:
            import google.generativeai as genai

            self.client = genai
            self.client.configure(api_key=self.api_key)
            self.generative_model = self.client.GenerativeModel(self.model_name)
            self.logger.info('Gemini SDK configured successfully with model %s.', self.model_name)
        except Exception as error:
            self.logger.warning('Failed to initialize Gemini SDK: %s', error)
            self.client = None
            self.generative_model = None

    def is_available(self) -> bool:
        return self.generative_model is not None and not self.disabled

    def generate(self, prompt: str, media_path: Optional[Path] = None) -> str:
        if not self.is_available():
            raise RuntimeError('Gemini SDK is not available.')

        try:
            self.logger.info('Gemini call')
            if media_path is not None and media_path.exists():
                response = self.generative_model.generate_content([prompt, str(media_path)])
            else:
                response = self.generative_model.generate_content(prompt)

            text = getattr(response, 'text', None)
            if text is None and hasattr(response, 'to_dict'):
                text = response.to_dict().get('text', '')
            if text is None:
                text = str(response)

            self.logger.debug('Raw Gemini response: %s', response)
            return text
        except Exception as error:
            error_text = str(error).lower()
            if 'quota exceeded' in error_text or 'resourceexhausted' in error_text or '429' in error_text:
                self.disabled = True
                if not self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.warning('Gemini quota exceeded. Falling back to Rule-Based Reasoner.')
                else:
                    self.logger.warning('Gemini quota exceeded. Falling back to Rule-Based Reasoner: %s', error)
            else:
                self.logger.debug('Gemini API request failed: %s', error)
            raise

    def _extract_text(self, response: Any) -> str:
        if response is None:
            return ''

        if isinstance(response, str):
            return response

        if hasattr(response, 'text'):
            return str(response.text)

        if isinstance(response, dict):
            if 'candidates' in response and response['candidates']:
                return str(response['candidates'][0].get('content', ''))
            if 'output' in response:
                return str(response['output'])

        if hasattr(response, 'to_dict'):
            data = response.to_dict()
            if 'candidates' in data and data['candidates']:
                return str(data['candidates'][0].get('content', ''))
            if 'output' in data:
                return str(data['output'])

        return str(response)

    def parse_json(self, text: str) -> dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError as error:
            self.logger.exception('Failed to parse JSON from Gemini response: %s', error)
            raise
