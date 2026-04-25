# Switch between user simulator versions here.
# v1 (passive/persona, uses passive_user.jinja / persona_user.jinja):
#   from .mimic_user import MimicUser
# v2 (feedback_mimic_user_v2.jinja — richer grounding model):
from .mimic_user_old import MimicUserV2 as MimicUser

__all__ = ["MimicUser"]
