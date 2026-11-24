# FieldCapture: AI Powered Hands Free Field Inspection Using Wearable Technology

## 1. Project Overview

FieldCapture is an AI powered field inspection platform designed to explore how wearable technology, computer vision and multimodal AI can transform property inspection workflows.

The project combines an intelligent inspection application with research and development into Meta AI glasses integration. Its objective is to enable inspectors to capture visual information, identify potential property damage, extract equipment details and generate structured inspection records through a hands free workflow.

The system brings together computer vision, machine learning, OCR, voice command processing, offline data handling and native iOS development.

## 2. Research Into Meta AI Glasses and Wearable Integration

A significant part of this project involved understanding how Meta AI glasses operate and how third party applications can access their capabilities.

Rather than treating the glasses as an ordinary Bluetooth camera, the research focused on the Meta Wearables Device Access Toolkit (DAT), its native application architecture and the communication requirements for accessing wearable device capabilities.

### Understanding the Wearable Architecture

The research explored the relationship between:

• Meta AI glasses as the wearable capture device.

• The Meta AI companion application for device setup and developer access.

• The Meta Wearables Device Access Toolkit for application integration.

• A native iOS application responsible for registration, permissions and device sessions.

• The computer vision backend responsible for processing captured inspection information.

The key architectural finding was that direct wearable camera integration requires a native application using Meta's supported SDK rather than relying exclusively on a browser based camera interface.

### Understanding the Device Access Toolkit

The project research covered the major SDK components:

**MWDATCore**

Responsible for wearable registration, device discovery, permissions, device selection and session management.

**MWDATCamera**

Provides camera capabilities, video streaming and access to captured image frames.

**MockDeviceKit**

Provides a development and testing pathway for simulating wearable device interactions without requiring physical glasses during early development.

### Researching the Connection Workflow

The intended connection workflow was mapped as follows:

1. Enable Developer Mode through the Meta AI application.
2. Configure the native iOS application.
3. Integrate the Meta Wearables SDK through Swift Package Manager.
4. Initialize the Wearables SDK.
5. Register the application with the wearable platform.
6. Discover and establish a device session.
7. Request the necessary camera permissions.
8. Initialize the camera stream.
9. Receive image frames for inspection processing.
10. Transfer captured information into the FieldCapture processing pipeline.

### Why Xcode Was Required

The research established that Meta's public Wearables Device Access Toolkit for iOS is distributed as a Swift Package and is designed for native iOS applications.

This led to the decision to introduce Xcode and Swift into the project architecture.

The native iOS component provides the intended bridge between the wearable device and the existing Python based inspection system.

The project includes an Xcode project scaffold, Swift connection layer, session management structure and application configuration files.

The next engineering stage is SDK integration, native compilation, connection verification and physical glasses testing.

## 3. AI and Computer Vision Research

The inspection processing pipeline was developed around the objective of converting visual property information into structured inspection records.

### Damage Detection

A machine learning workflow was developed for binary damage presence detection.

The approach combines:

• OpenCV based image feature extraction.

• Random Forest classification.

• Confidence based prediction thresholds.

• Conservative handling of uncertain predictions.

• Technician review for preliminary damage subtype suggestions.

The resulting model provides an initial foundation for automated damage screening.

### Nameplate OCR and Equipment Recognition

The OCR subsystem explores a dual engine recognition approach using PaddleOCR and Tesseract.

The processing workflow includes:

1. Image preprocessing.
2. Text detection and recognition.
3. Geometric and textual alignment of OCR outputs.
4. Field extraction and normalization.
5. Identifier plausibility checks.
6. Review of ambiguous equipment information.

This approach aims to improve the reliability of equipment identification and structured nameplate extraction.

### Image Quality Assessment

An image quality subsystem evaluates captured images and provides retake guidance when image quality is insufficient for further processing.

### Inspection Command Processing

The application supports voice and typed inspection instructions.

Example commands:

```text
capture nameplate
capture damage kitchen north wall
note: subfloor soft
```

The command processing layer translates natural language instructions into structured inspection actions.

## 4. Promising Development Results

The development and evaluation work has produced encouraging preliminary results.

| Research component | Development result |
|---|---:|
| Nameplate OCR exact field recognition | 92% |
| Damage presence model overall accuracy | 90.05% |
| Selective damage prediction accuracy | 95.83% |
| Automated software unit tests | 57 passing |

The OCR results demonstrate promising equipment information extraction capabilities.

The damage detection experiments indicate that confidence based prediction and conservative decision thresholds can improve the reliability of accepted predictions.

The selective prediction approach is particularly relevant to inspection workflows because uncertain cases can be directed for technician review rather than automatically treated as confirmed findings.

These results provide a foundation for further model development and evaluation under practical inspection conditions.

## 5. Application Architecture

The project combines the following components:

**Frontend**

Browser based inspection interface supporting camera access, image uploads, voice commands, inspection results and editable fields.

**Python processing backend**

Coordinates inspection commands, image processing, OCR, damage inference and structured record generation.

**Machine learning layer**

Provides image feature extraction, trained damage presence inference and confidence based decision handling.

**Native iOS layer**

Provides the Xcode project structure and exploratory Meta Wearables Device Access Toolkit integration.

**Data management layer**

Supports offline queuing, retry handling, duplicate prevention, JSON validation and estimate line item mapping.

## 6. Technology Stack

| Area | Technologies |
|---|---|
| Programming | Python, Swift, JavaScript |
| Backend | Python HTTP application |
| Frontend | HTML, CSS, JavaScript |
| Computer Vision | OpenCV |
| Machine Learning | Scikit learn, Random Forest |
| OCR | PaddleOCR, Tesseract |
| Mobile Development | Swift, SwiftUI, Xcode |
| Wearable Integration | Meta Wearables Device Access Toolkit |
| Data Processing | JSON, structured validation |
| Testing | Python unittest |

## 7. Repository Structure

```text
fieldcapture/
│
├── docs/
│
├── eval/
│
├── fieldcapture/
│   ├── commands.py
│   ├── quality.py
│   ├── ocr.py
│   ├── ocr_backend.py
│   ├── damage.py
│   ├── extract.py
│   ├── outbox.py
│   ├── process.py
│   ├── validate.py
│   └── lineitems.py
│
├── ios/
│   └── FieldCapture.xcodeproj/
│
├── models/
│   └── damage_presence_rf.pkl
│
├── reference_images/
├── tests/
├── tools/
├── web/
│   └── app.html
│
├── app.py
├── requirements.txt
└── README.md
```

## 8. Installation and Execution

### Requirements

• Python 3.11 or newer.

• Tesseract OCR.

• Google Chrome or Microsoft Edge.

### Windows Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Open the application at:

```text
http://localhost:8000
```

### Run Automated Tests

```powershell
python -m compileall -q app.py fieldcapture eval tests
python -m unittest discover -s tests -v
```

## 9. Native iOS Development Requirements

The current Meta Wearables iOS SDK documentation specifies:

• macOS with a compatible Xcode installation.

• Xcode 26.4 or newer for the current public sample applications.

• Swift 6.3.

• iOS 17.2 or newer deployment target.

• Meta AI companion application.

• Supported Meta AI glasses or MockDeviceKit for development.

• Developer Mode enabled through the Meta AI application.

The SDK is integrated through Swift Package Manager:

```text
https://github.com/meta/meta-wearables-dat-ios
```

The principal packages for camera integration are:

```text
MWDATCore
MWDATCamera
```

The native development process involves application registration, device session creation, permission handling, camera streaming and image processing integration.

## 10. Future Research and Engineering Direction

The next development phase focuses on:

1. Completing and compiling the native iOS application.
2. Integrating and verifying the Meta Wearables SDK.
3. Establishing a connection with supported Meta AI glasses.
4. Testing live camera streaming and image capture.
5. Connecting wearable image frames to the FieldCapture AI processing pipeline.
6. Improving damage detection and OCR performance.
7. Evaluating the system through practical inspection sessions.
8. Measuring capture latency, processing time and connection reliability.

## 11. Conclusion

FieldCapture explores the intersection of wearable computing, artificial intelligence and computer vision for practical property inspection.

The project combines an implemented inspection software prototype with research into Meta's wearable device architecture and an exploratory native iOS integration.

The preliminary AI development results are promising and provide a starting point for further experimentation.

The broader research objective is to move towards an integrated hands free inspection workflow in which wearable devices capture visual information and AI systems assist with damage screening, equipment recognition and structured inspection documentation.
