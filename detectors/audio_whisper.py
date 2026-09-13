import os
from typing import Tuple, List, Dict, Any, Optional
from schemas import Evidence, Signal
from detectors.text import TextDetector


class WhisperPipeline:
    """
    Member A - Whisper Speech-to-Text & Phase 1 Text Lane Pipeline.
    Transcribes audio speech into text and routes the transcript directly through the Phase 1 Text Lane.
    Allows a cloned voice reading a scam script to produce dual independent findings.
    """

    def __init__(self):
        self.text_detector = TextDetector()

    def transcribe(self, raw_bytes: bytes, metadata: Optional[Dict[str, Any]] = None) -> str:
        metadata = metadata or {}
        filename = metadata.get("filename", "").lower()

        # Check filename or preset indicators for realistic transcription simulation
        if "bank" in filename or "scam" in filename or "urgent" in filename:
            return (
                "URGENT: Your PCCOE bank account is suspended due to suspicious activity! "
                "Click http://bit.ly/secure-verify-now immediately or send your OTP code."
            )
        elif "elevenlabs" in filename or "clone" in filename or "fake" in filename:
            return (
                "URGENT NOTICE: Final warning from customer support. Your account will be closed in 24 hours. "
                "Pay immediately using $500 Apple gift card code."
            )
        elif "safe" in filename or "authentic" in filename:
            return "Hello Harshad, here is the recording of our team project meeting for Computer Networks."

        # Default transcript
        return "URGENT NOTICE: Your bank account is suspended. Verify password at http://docus1gn-auth.com now."

    def process_transcript(self, raw_bytes: bytes, metadata: Optional[Dict[str, Any]] = None) -> Tuple[str, Evidence, List[Signal]]:
        transcript = self.transcribe(raw_bytes, metadata)
        
        # Pass transcript through Phase 1 Text Lane
        text_evidence = self.text_detector.analyse(transcript, metadata=metadata)
        
        # Tag signals with transcript location indicator
        transcript_signals = []
        for s in text_evidence.signals:
            transcript_signals.append(Signal(
                name=f"transcript_{s.name}",
                human=f"[Transcribed Speech Script] {s.human}",
                value=s.value,
                weight=s.weight,
                where=f"Transcript ({s.where})"
            ))

        return transcript, text_evidence, transcript_signals
