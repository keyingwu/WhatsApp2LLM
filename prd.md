# Whatsapp Chat to LLM-Ready Text Converter (PRD)

## 1. Introduction

### 1.1 Purpose

The purpose of this document is to outline the requirements for developing an application that converts WhatsApp chat exports (ZIP files) into LLM-ready text files. The application will handle the conversion and transcription of audio files within the chat history, integrating the transcribed text back into the conversation seamlessly. This tool aims to facilitate the preparation of chat data for analysis by Large Language Models (LLMs).

### 1.2 Background

WhatsApp allows users to export chat histories, which often include audio messages in the `.opus` format. Currently, processing these exports for use with LLMs is cumbersome due to the presence of audio files that need manual transcription. This application addresses this pain point by automating the transcription and integration process using Python, OpenAI's Whisper for speech-to-text, and Gradio for the user interface.

### 1.3 Example ZIP File Structure

To better understand the input the application will handle, here is an example of a typical WhatsApp chat export ZIP file structure:

```
.
├── 00000001-AUDIO-2024-11-22-14-35-49.opus
├── 00000002-AUDIO-2024-11-22-14-36-28.opus
└── _chat.txt
```

The `_chat.txt` file contains the chat history, which may include references to audio attachments. An excerpt from `_chat.txt`:

```
[08.01.17 22:20:00] Person A: Hello!
‎[22.11.24 14:35:49] Person B:‎<附件：00000001-AUDIO-2024-11-22-14-35-49.opus>
‎[22.11.24 14:36:28] Person A:‎<附件：00000002-AUDIO-2024-11-22-14-36-28.opus>
```

**Note**: In this example, the term "附件" is the Chinese word for "Attachment". The application should be compatible with chat exports from systems using different languages and character encodings.

## 2. Scope

### 2.1 In Scope

- Processing WhatsApp chat export ZIP files.
- Converting audio files from `.opus` to a standard audio format (e.g., `.wav`).
- Transcribing audio files into text using OpenAI's Whisper.
- Inserting transcribed text back into the chat history at the correct position.
- Supporting multiple languages and character encodings.
- Handling various language-specific labels for attachments (e.g., "附件" in Chinese).
- Providing a simple user interface using Gradio.

### 2.2 Out of Scope

- Editing or modifying non-audio content within the chat history.
- Translating chat content from one language to another.
- Processing media files other than audio (e.g., images, videos).
- Real-time processing of ongoing chats (only exported chats are supported).

## 3. Functional Requirements

### 3.1 File Import

- **FR1**: The application shall accept a WhatsApp chat export ZIP file as input through a Gradio-based user interface.

### 3.2 Audio File Conversion

- **FR2**: The application shall identify audio files in the `.opus` format within the ZIP file.
- **FR3**: The application shall convert `.opus` audio files to `.wav` format using FFmpeg.

### 3.3 Audio Transcription

- **FR4**: The application shall transcribe audio files into text using OpenAI's Whisper speech-to-text engine.
- **FR5**: The transcription service shall support multiple languages as per the content of the audio files.

### 3.4 Chat History Processing

- **FR6**: The application shall parse the `_chat.txt` file and identify references to audio attachments.
- **FR7**: The application shall replace audio attachment references in the `_chat.txt` file with the transcribed text.
- **FR8**: The application shall maintain the original formatting and timestamps of the chat history.

### 3.5 Multi-Language Support

- **FR9**: The application shall recognize and process attachment labels in different languages (e.g., "附件", "Attachment").
- **FR10**: The application shall correctly handle character encodings (e.g., UTF-8) to support non-Latin scripts.

### 3.6 Output Generation

- **FR11**: The application shall generate a single text file containing the processed chat history, ready for LLM input.
- **FR12**: The application shall provide options for output formats if necessary (e.g., plain text, JSON).

### 3.7 User Interface

- **FR13**: The application shall provide a simple web-based user interface using Gradio for users to select input files and configure options.
- **FR14**: The application shall display progress and status messages during processing within the UI.

## 4. Non-Functional Requirements

### 4.1 Performance

- **NFR1**: The application should process files efficiently, with reasonable performance for large chat histories.

### 4.2 Scalability

- **NFR2**: The application should handle chat exports of varying sizes without significant degradation in performance.

### 4.3 Reliability

- **NFR3**: The application should handle errors gracefully and provide meaningful error messages to the user.

### 4.4 Usability

- **NFR4**: The application should be user-friendly and not require advanced technical knowledge to operate.

### 4.5 Compatibility

- **NFR5**: The application should run on major operating systems (e.g., Windows, macOS, Linux).

### 4.6 Security

- **NFR6**: The application should securely handle user data, especially since Whisper operates offline, enhancing privacy.

### 4.7 Maintainability

- **NFR7**: The application code should be well-documented to facilitate future updates and maintenance.

## 5. System Architecture Overview

### 5.1 High-Level Components

1. **Input Module**: Handles the import of the ZIP file and extraction of contents.
2. **Audio Processing Module**:
   - Converts `.opus` files to `.wav` format using FFmpeg.
   - Transcribes audio files into text using OpenAI's Whisper.
3. **Chat Parser Module**:
   - Parses the `_chat.txt` file.
   - Identifies and replaces audio attachment references with transcribed text.
4. **Output Module**: Generates the final LLM-ready text file.
5. **User Interface Module**: Provides a web-based interface using Gradio.
6. **Language Support Module**: Detects and handles multiple languages in attachment labels and content.

### 5.2 Data Flow

1. **User Interface Module**: User uploads the WhatsApp chat export ZIP file via Gradio.
2. **Input Module**: Extracts `_chat.txt` and audio files.
3. **Chat Parser Module**: Scans `_chat.txt` for audio attachment references.
4. **Audio Processing Module**:
   - Converts `.opus` files to `.wav` format.
   - Transcribes audio files using Whisper.
5. **Chat Parser Module**: Receives transcribed text and replaces attachment references.
6. **Output Module**: Compiles the processed chat history into a text file.
7. **User Interface Module**: Provides the LLM-ready text file for the user to download.

## 6. Technical Considerations

### 6.1 Programming Language

- **Python 3.x**

  - Chosen for its simplicity, extensive library support, and rapid development capabilities.

### 6.2 Audio Conversion

- **FFmpeg**

  - Utilize FFmpeg to convert `.opus` files to `.wav` format.
  - Control FFmpeg via Python using the `subprocess` module or a wrapper like `ffmpeg-python`.

### 6.3 Speech-to-Text Service

- **OpenAI's Whisper**

  - Install via `pip install -U openai-whisper`.
  - Supports multiple languages and operates offline.
  - Enhances privacy by keeping data local.

### 6.4 Chat Parsing and Text Processing

- **Regular Expressions**

  - Use Python's built-in `re` module to identify attachment references in `_chat.txt`.

- **Character Encoding**

  - Ensure all file operations use UTF-8 encoding to support international characters.

### 6.5 User Interface

- **Gradio**

  - Provides a simple web-based UI for file upload and status display.
  - Quick to set up with minimal code.
  - Suitable for MVP development.

### 6.6 Language Detection (Optional)

- **Language Detection Libraries**

  - Use `langdetect` or similar libraries if needed for selecting transcription models.

### 6.7 Timezone and Date Formats

- Handle different date and time formats that may appear in `_chat.txt`.
- Maintain the original timestamps to preserve conversation flow.

## 7. Error Handling

### 7.1 Missing Files

- If an audio file referenced in `_chat.txt` is missing, log a warning and skip the transcription for that file.

### 7.2 Transcription Failures

- If transcription fails for a file, insert a placeholder message in the chat history indicating the failure.

### 7.3 Unsupported Formats

- If the ZIP file contains unsupported file types, ignore them without interrupting the process.

### 7.4 Invalid Input

- Validate the ZIP file structure and contents before processing.
- Provide clear error messages via the Gradio UI if the input file is invalid.

## 8. Future Enhancements

- **FE1**: Support transcription of audio files in other formats (e.g., voice notes in different formats).
- **FE2**: Add translation capabilities to translate chat content into a target language.
- **FE3**: Enhance the user interface with additional features or migrate to a more customizable framework if needed.
- **FE4**: Integrate processing of other media types (e.g., images with OCR).

## 9. Assumptions and Dependencies

- Users have the necessary permissions to access and process the chat export files.
- Whisper supports the languages present in the audio files.
- FFmpeg is installed and accessible from the system where the application runs.
- The application can operate offline, enhancing privacy and data security.

## 10. Open Questions

- **OQ1**: What measures will be taken to handle resource-intensive operations, especially with larger Whisper models?
- **OQ2**: Will users have the option to select different Whisper models based on their hardware capabilities?
- **OQ3**: How will the application ensure compatibility with various date and time formats in `_chat.txt`?

---

**Note**: This updated PRD incorporates the use of Python, OpenAI's Whisper for speech-to-text transcription, and Gradio for the user interface. These choices aim to streamline development and provide a robust, user-friendly application that operates efficiently and respects user privacy.

---