def safety_triage(text):

    text = text.lower()

    red_flags = [
        "unconscious",
        "seizure",
        "respiratory distress",
        "cardiac arrest",
        "coma",
        "not breathing"
    ]

    high_risk_exposures = [
        "pesticide",
        "organophosphate",
        "poison",
        "chemical ingestion",
        "unknown substance",
        "toxic exposure"
    ]

    for flag in red_flags:
        if flag in text:
            return {
                "risk_level": "CRITICAL",
                "reason": f"Life-threatening symptom detected: {flag}",
                "action": "Immediate emergency hospital referral"
            }

    for exposure in high_risk_exposures:
        if exposure in text:
            return {
                "risk_level": "HIGH",
                "reason": f"High-risk toxic exposure detected: {exposure}",
                "action": "Urgent emergency evaluation required"
            }

    return {
        "risk_level": "MODERATE",
        "reason": "No life-threatening symptoms detected",
        "action": "Monitor patient and provide guidance"
    }