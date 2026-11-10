# Laya: Gnani Internship Challenge Submission Draft

## 1. Submission Text (170 words)

When people practice a speech or presentation, they usually cannot tell which specific seconds sounded flat, rushed, or hesitant. Typical speech feedback tools only provide high-level text impressions without pointing to the exact physical moments in the audio where the delivery slipped.

Laya solves this by measuring acoustic features directly from the audio recording: fundamental frequency (F0 pitch), speaking rate, and pauses. It flags objective flaw intervals using frozen statistical deviation thresholds (|z| >= 2.0) and scores delivery objectively against an explainable rubric.

For the challenge, we integrated Gnani Prisma v3.0 STT and Timbre v3.0 TTS to support Indian languages. Prisma transcribes Hindi and Hinglish speech in chunks of 25 seconds or less, allowing pitch variation checks on vernacular recordings. For each flagged region, Prisma transcribes the local audio slice, and Timbre synthesizes fixed coaching tips so speakers can hear model delivery ("Model voice, not your voice").

Current limits are straightforward: Prisma provides chunk-level timing rather than word timestamps, so pace and pause checks remain English-only. The system was tested on two clips per language from one speaker.

## 2. 60-Second Demo Script

### 00:00 - 00:08 (8 seconds): The Problem
- **Visual**: Screen shows the Laya local dashboard header and problem statement.
- **Voiceover**: "When you practice a speech, it is hard to tell which exact seconds sounded flat or rushed. Most feedback tools only give vague text opinions."

### 00:08 - 00:20 (12 seconds): Hindi Recording Analysis
- **Visual**: In the recording card, select the "Hindi" language pill. Upload `scratch/test_hi.wav` and click "Run Analysis". The timeline appears with pitch contours and a flagged flaw region. [TO VERIFY ON SCREEN]
- **Voiceover**: "Laya measures physical sound waves directly. Here on a Hindi recording, it transcribes with Gnani Prisma v3.0 and flags a monotone section where the speaker's pitch stayed flat at a z-score of minus 2.5."

### 00:20 - 00:32 (12 seconds): "What you said here" and "Hear it fixed"
- **Visual**: Click on the flagged flaw region in the table. Under the flag, the local excerpt text appears. Click the "Hear it fixed" button. The audio player loads with the explicit badge "Model voice, not your voice". [TO VERIFY ON SCREEN]
- **Voiceover**: "Clicking the flag cuts the audio around that phrase and shows what was said. The 'Hear it fixed' button calls Gnani Timbre to read out a fixed practice tip in Hindi, clearly labelled as a model voice."

### 00:32 - 00:48 (16 seconds): Retake Comparison
- **Visual**: Click "Record a retake". The banner shows "Retake for: test_hi". Load the second take and click "Run Analysis". The "Take 1 and take 2" card shows rubric score deltas and flag tracking ("not flagged in take 2"). [TO VERIFY ON SCREEN]
- **Voiceover**: "When you record a retake, Laya compares both takes side by side. It shows whether flagged seconds decreased and tells you whether prior flaws are still flagged or not flagged in take 2."

### 00:48 - 01:00 (12 seconds): Summary and Limits
- **Visual**: Scroll over the delivery score rubric, the script mix bar, and documentation links.
- **Voiceover**: "Laya gives speakers objective, reproducible measurements. Prisma and Timbre extend that foundation to Hindi and Hinglish, with clear boundaries on what was measured."

## 3. Social Post Draft (LinkedIn / X)

Most speech coaching tools give vague advice because they cannot hear your voice. We built Laya to measure speech directly: pitch, speaking speed, and pauses against statistical baselines.

For the Great Indian AI Internship Challenge, we added support for Hindi and Hinglish using Gnani Prisma v3.0 for speech transcription and Timbre v3.0 for spoken coaching tips. Users can see the exact seconds where their voice went flat, hear a model voice read fixed practice tips, and compare two takes to track changes.

Code and evaluation on GitHub.

#GreatIndianAIInternshipChallenge #GnaniAI

## 4. Submission Checklist

- [ ] Challenge terms on reusing an existing project [NOT CONFIRMED]
- [ ] Submission deadline [UNKNOWN]
- [ ] Browser test of Hindi flat vs normal retake [NOT RUN]
- [ ] GitHub repository link: https://github.com/rahul-dev-cmd/Laya.git (branch `gnani`)
- [ ] Dataset public download link [NOT DONE]

