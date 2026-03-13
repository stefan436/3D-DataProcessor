# 🏔️ 3D-DataProcessor

![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![GUI](https://img.shields.io/badge/GUI-PyQt6-orange)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)

**3D-DataProcessor** ist eine leistungsstarke, GUI-gestützte Python-Anwendung zur Verarbeitung, Transformation und Visualisierung von 3D-Geodaten. Die Software überführt Rohdaten wie GeoTIFFs und LiDAR-Punktwolken in wasserdichte 3D-Meshes - ideal für den 3D-Druck oder zur Erstellung von Modellen.

## 📋 Inhaltsverzeichnis
1. [Features](#-features)
2. [Beispieldaten](#-beispieldaten)
3. [Installation](#-installation)
4. [Nutzungshinweise](#-nutzungshinweise)
5. [Technologie-Stack](#-technologie-stack)

## ✨ Features

* **Universeller Datenimport:** Unterstützung für GeoTIFF, LAZ, ASCII-Zip und DOM+DOP (Digitale Oberflächenmodelle inkl. Orthofotos).
* **Geodaten-Processing:** Bounding-Box Cropping, CRS-Transformationen und Z-Überhöhung.
* **Intelligentes Meshing:** Erstellt 3D-Modelle über Screened Poisson Reconstruction oder den Ball-Pivoting Algorithm (BPA) (für LiDAR Punktwolken). Bietet Optionen für HC-Smoothing und Polygon-Reduktion (Decimation).
* **Post-Processing für 3D-Druck:** Automatische Generierung von Wänden und soliden Böden, um offene Geländemodelle in druckbare, wasserdichte Volumenkörper zu verwandeln.
* **Integrierte Visualisierung:** Betrachte Daten direkt in der App über interaktive 2D-Plots, 3D-Scatter-Views oder 3D-Surface-Renderings.

## 🗺️ Beispieldaten

Beispieldaten für dieses Projekt können unter https://geodaten.bayern.de/opengeodata/ heruntergeladen werden. Nutze hierfür bevorzugt die DGM (Digitale Geländemodelle) oder DOM (Digitale Oberflächenmodelle).

## 🚀 Installation

Es wird empfohlen, das Projekt in einer virtuellen Python-Umgebung zu installieren, um Konflikte mit anderen Systempaketen zu vermeiden.

**1. Repository klonen:**  
`git clone [https://github.com/stefan436/3D-DataProcessor.git](https://github.com/stefan436/3D-DataProcessor.git)`  
`cd 3D-DataProcessor`

**2. Virtuelle Umgebung erstellen und aktivieren:**
#### Windows
`python -m venv venv`  
`venv\Scripts\activate`

#### Linux/macOS
`python3 -m venv venv`  
`source venv/bin/activate`

**3. Abhängigkeiten installieren:**  
`pip install -r requirements.txt`

## 💻 Nutzungshinweise

Starte die grafische Benutzeroberfläche über das Hauptskript aus dem Hauptverzeichnis heraus:  
`python src/main.py`

#### Workflow in der App:

1. Wähle im oberen Bereich der App ein **Input-Verzeichnis** und ein **Output-Verzeichnis**.

2. Navigiere durch die Tabs von links nach rechts:

   - **1. IMPORT:** Konvertiere deine `.tif` oder `.laz` Dateien in das interne `.npz` Format.
   - **2. PROCESSING:** Schneide die Daten zu oder füge eine Z-Überhöhung hinzu.
   - **3. MESHING:** Wandle die Daten in `.ply` 3D-Modelle um und generiere bei Bedarf Wände und Boden; exportiere als `.stl`.
   - **4. ANSICHT:** Überprüfe deine Zwischen- oder Endergebnisse visuell.

## 🛠️ Technologie-Stack

Dieses Projekt baut auf Open-Source-Bibliotheken auf:

- **UI:** PyQt6  
- **Data Processing:** NumPy, SciPy, Pandas, Rasterio, Laspy  
- **3D & Meshing:** Open3D, PyMeshLab, Trimesh  
- **Visualisierung:** Vispy, Matplotlib