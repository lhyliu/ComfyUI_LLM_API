from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderProfile:
    profile: str
    base_url: str
    supports_images: bool

    def as_dict(self):
        return {
            "profile": self.profile,
            "base_url": self.base_url,
            "supports_images": self.supports_images,
        }


PROVIDER_PROFILE_NAMES = (
    "custom_text_image",
    "custom_text_only",
    "openai",
    "deepseek",
    "qwen_dashscope_cn",
    "doubao_ark_cn",
    "zhipu_glm",
    "minimax_global",
)


_PROVIDER_PROFILES = {
    "custom_text_image": ProviderProfile("custom_text_image", "", True),
    "custom_text_only": ProviderProfile("custom_text_only", "", False),
    "openai": ProviderProfile("openai", "https://api.openai.com/v1", True),
    "deepseek": ProviderProfile("deepseek", "https://api.deepseek.com", False),
    "qwen_dashscope_cn": ProviderProfile(
        "qwen_dashscope_cn",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
        True,
    ),
    "doubao_ark_cn": ProviderProfile(
        "doubao_ark_cn",
        "https://ark.cn-beijing.volces.com/api/v3",
        True,
    ),
    "zhipu_glm": ProviderProfile(
        "zhipu_glm",
        "https://open.bigmodel.cn/api/paas/v4",
        True,
    ),
    "minimax_global": ProviderProfile(
        "minimax_global",
        "https://api.minimax.io/v1",
        False,
    ),
}


def resolve_provider_profile(provider_profile, api_baseurl):
    if provider_profile not in _PROVIDER_PROFILES:
        raise ValueError(f"Unknown provider profile: {provider_profile}")

    profile = _PROVIDER_PROFILES[provider_profile]
    base_url = (api_baseurl or "").strip() or profile.base_url
    return ProviderProfile(profile.profile, base_url, profile.supports_images).as_dict()
