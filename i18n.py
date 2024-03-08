import gettext
from openpilot.common.params import Params

locale_dir = "/data/openpilot/selfdrive/assets/locales"
# supported_language = ["en-US", "zh-TW", "zh-CN", "ja-JP", "ko-KR"]
supported_languages = {
  "main_en": "en-US",
  "main_zh-CHT": "zh-TW",
  "main_zh-CHS": "zh-CN",
  "main_ko": "ko-KR",
  "main_ja": "ja-JP",
}

def events():
  import sys
  try:
    if sys.argv[0] == "./manager.py":
      lang = Params().get("LanguageSetting", encoding='utf8')
    else:
      lang = "main_en"
    locale = supported_languages[lang.strip()]
  except:
    locale = "en-US"
  i18n = gettext.translation("events", localedir=locale_dir, fallback=True, languages=[locale])
  i18n.install()
  return i18n.gettext
