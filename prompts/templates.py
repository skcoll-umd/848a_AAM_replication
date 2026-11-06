"""
Prompt templates for the AAM replication pipeline.

Exact prompts from Palta et al. (2026), Appendix A.1–A.6.
"""

LIKERT_SCALE = (
    "Likert Scale:\n"
    "1 - Impossible\n"
    "2 - Technically Possible\n"
    "3 - Plausible\n"
    "4 - Likely\n"
    "5 - Very Likely"
)


def rationale_pro(question: str, choice: str) -> str:
    """Step 1 (Prompt A.1) — generate a PRO rationale supporting `choice`."""
    return (
        f"Give your best rationale as to why the given choice is a plausible answer "
        f"for the question. The rationale should not be more than 2 sentences long and "
        f"should be concise. Your rationale has to be in favour of the plausibility of "
        f"the choice, no matter what. Some of these situations may require creativity or "
        f"thinking outside the box to come up with a rationale that best explains the "
        f"plausibility of the answer choice. Feel free to be as creative as required.\n"
        f"Question: {question}\n"
        f"Choice: {choice}\n"
        f"Generated Rationale:"
    )


def rationale_con(question: str, choice: str) -> str:
    """Step 1 (Prompt A.2) — generate a CON rationale arguing against `choice`."""
    return (
        f"Give your best rationale as to why the given choice is an implausible answer "
        f"for the question. The rationale should not be more than 2 sentences long and "
        f"should be concise. Your rationale has to be in favour of the implausibility of "
        f"the choice, no matter what. Some of these situations may require creativity or "
        f"thinking outside the box to come up with a rationale that best explains the "
        f"implausibility of the answer choice. Feel free to be as creative as required.\n"
        f"Question: {question}\n"
        f"Choice: {choice}\n"
        f"Generated Rationale:"
    )


def rating_no_rationale(question: str, choice: str) -> str:
    """Step 2 (Prompt A.3) — baseline condition: rate plausibility with no rationale."""
    return (
        f"I will give you a question and an answer choice. Your task is to rate the "
        f"plausibility of the answer choice on the following Likert scale:\n"
        f"{LIKERT_SCALE}\n"
        f"Your answer should just be a complete option from the Likert scale. Nothing "
        f"else. No extra words or explanations.\n"
        f"Question: {question}\n"
        f"Choice: {choice}\n"
        f"Generated Rating:"
    )


def rating_pro_only(question: str, choice: str, pro: str) -> str:
    """Step 2 (Prompt A.4) — PRO condition: rate after seeing the PRO rationale."""
    return (
        f"I will give you a question, an answer choice, and a rationale that best "
        f"explains the plausibility of the answer choice. Your task is to rate the "
        f"plausibility of the answer choice on the following Likert scale taking into "
        f"consideration the rationale given.\n"
        f"{LIKERT_SCALE}\n"
        f"Your answer should just be a complete option from the Likert scale. Nothing "
        f"else. No extra words or explanations.\n"
        f"Question: {question}\n"
        f"Choice: {choice}\n"
        f"Explanation: {pro}\n"
        f"Generated Rating:"
    )


def rating_con_only(question: str, choice: str, con: str) -> str:
    """Step 2 (Prompt A.5) — CON condition: rate after seeing the CON rationale."""
    return (
        f"I will give you a question, an answer choice, and a rationale that best "
        f"explains the implausibility of the answer choice. Your task is to rate the "
        f"plausibility of the answer choice on the following Likert scale taking into "
        f"consideration the rationale given.\n"
        f"{LIKERT_SCALE}\n"
        f"Your answer should just be a complete option from the Likert scale. Nothing "
        f"else. No extra words or explanations.\n"
        f"Question: {question}\n"
        f"Choice: {choice}\n"
        f"Explanation: {con}\n"
        f"Generated Rating:"
    )


def rating_pro_con(question: str, choice: str, pro: str, con: str) -> str:
    """Step 2 (Prompt A.6) — PRO+CON condition: rate after seeing both rationales."""
    return (
        f"I will give you a question, an answer choice, a rationale that best explains "
        f"the plausibility of the answer choice, and a rationale that best explains the "
        f"implausibility of the answer choice. Your task is to rate the plausibility of "
        f"the answer choice on the following Likert scale taking into consideration the "
        f"rationales given.\n"
        f"{LIKERT_SCALE}\n"
        f"Your answer should just be a complete option from the Likert scale. Nothing "
        f"else. No extra words or explanations.\n"
        f"Question: {question}\n"
        f"Choice: {choice}\n"
        f"Explanation 1: {pro}\n"
        f"Explanation 2: {con}\n"
        f"Generated Rating:"
    )
