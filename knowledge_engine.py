"""
Campus Knowledge Engine for DROBOT (Jaypee University of Information Technology)
Loads, indexes, and retrieves campus facts for real-time voice queries with sub-millisecond latency.
"""

import os
import re
import math
from collections import Counter
from typing import List, Dict, Any, Tuple

# Core curated facts about JUIT Waknaghat for instant grounding
CAMPUS_CORE_SUMMARY = """
Jaypee University of Information Technology (JUIT) is located in Waknaghat, Solan District, Himachal Pradesh on a scenic 25-acre hill campus.
Established in 2002 via HP Act No. 14, it is the first State Private University in Himachal Pradesh.
Accreditation: NAAC A+ Grade (CGPA 3.40), NBA accredited for B.Tech Biotech, NIRF 201-300 band, ranked 6th among India's Best Private Technical Universities by Outlook-ICARE.
Key Leadership:
- Chancellor: Governor of Himachal Pradesh (Sh. Kavinder Gupta)
- Pro-Chancellor: Sh. Manoj Gaur (Executive Chairman, Jaypee Group)
- CEO: Sh. Manu Bhaskar Gaur
- Vice Chancellor: Prof. (Dr.) Rajendra Kumar Sharma
- Registrar & Dean of Students: Brigadier RK Sharma, SM (Retd.)
- Dean Academics & Research: Prof. (Dr.) Ashok Kumar Gupta
- Dean Accreditations & CoE: Prof. (Dr.) Sunil Kumar Khah

Departments & HODs:
- Computer Science & IT (CSE/IT): HOD Prof. (Dr.) Vivek Sehgal (AI/ML, Data Science, Cyber Security, Full Stack)
- Electronics & Communication (ECE): HOD Prof. (Dr.) Rajiv Kumar
  * ECL 5 is the Drobotics Lab / e-Yantra Robotics Lab
  * ECL 7 is the Embedded Systems & Robotics Lab
- Biotechnology & Bioinformatics: HOD Prof. (Dr.) Sudhir Kumar
- Civil Engineering: HOD Prof. (Dr.) Ashish Kumar
- Humanities & Social Sciences / Jaypee Business School (JBSW): HOD Prof. (Dr.) Amit Srivastava (BBA, MBA)
- Mathematics: HOD Prof. (Dr.) Rakesh Kumar Bajaj
- Physics & Materials Science: HOD Prof. (Dr.) P.B. Barman

Admissions & Fees:
- B.Tech Admission: 50% seats through JEE Main All India Rank, 50% through 10+2 PCM merit.
- Tuition: ~Rs. 1,60,500 per semester; Hostel: ~Rs. 80,000 per semester.
- Defense ward rebate: 30% to 35%.
Campus Amenities:
- FSSAI 'Eat Right Campus' certified messes, dispensary with 24/7 medical support, gym, sports grounds, laundry, on-campus bank/ATM.

D-Bot Robot Hardware & Specifications:
- Name: D-Bot (autonomous service and delivery robot on wheels at JUIT Waknaghat).
- Main Onboard Compute / Brain: NVIDIA Jetson Orin Nano Super Dev Kit (running ROS 2, audio capture/playback, motor control).
- Depth Camera / Vision: Intel RealSense D435i depth camera (RGB video + active stereo IR depth + integrated IMU).
- 2D LiDAR / Scanner: Slamtec RPLiDAR A1 M8 (360-degree laser range scanner for SLAM mapping and obstacle avoidance).
- Locomotion / Chassis: Omnidirectional wheels (holonomic chassis capable of forward/backward motion, sideways strafing, diagonal movement, and 360-degree rotation in place).
- Development Lab: Drobotics Lab / e-Yantra Lab in ECL 5 (Department of ECE, JUIT Waknaghat).
"""

class KnowledgeEngine:
    def __init__(self, data_path: str = "/home/thor/dbot_work/clean_research.txt"):
        self.data_path = data_path
        self.chunks: List[Dict[str, Any]] = []
        self._load_and_chunk()
        self._build_index()

    def _load_and_chunk(self):
        # 1. Base structured knowledge chunks
        base_sections = [
            {
                "id": "dbot_hardware",
                "title": "D-Bot Robot Identity, Hardware Specifications, Sensors, and Chassis",
                "keywords": [
                    "dbot", "d-bot", "robot", "hardware", "spec", "specs", "specifications",
                    "orin", "nano", "super", "dev kit", "jetson", "compute", "processor", "brain",
                    "camera", "depth camera", "d435i", "intel", "realsense", "lidar", "rplidar",
                    "a1", "m8", "laser", "sensor", "sensors", "wheel", "wheels", "omnidirectional",
                    "omni", "mecanum", "holonomic", "strafe", "movement", "chassis", "built", "components"
                ],
                "content": (
                    "D-Bot Robot Hardware Specifications & Architecture: "
                    "- Main Onboard Compute: NVIDIA Jetson Orin Nano Super Dev Kit, which handles local sensor acquisition, motor control, and real-time audio communication. "
                    "- Vision & Depth Perception: Intel RealSense D435i depth camera providing high-resolution RGB video, active stereo infrared depth sensing, and an integrated IMU. "
                    "- Laser Range Scanner (LiDAR): Slamtec RPLiDAR A1 M8 360-degree 2D laser range scanner (up to 12-meter range, 8000 samples/sec) used for SLAM mapping, localization, and obstacle avoidance. "
                    "- Drive & Locomotion: Omnidirectional wheels (holonomic drive), enabling D-Bot to move forward, backward, strafe directly sideways left or right, slide diagonally, and spin 360 degrees in place. "
                    "- Research & Development: Built and programmed at Drobotics Lab / e-Yantra Lab in ECL 5, Department of Electronics and Communication Engineering (ECE), JUIT Waknaghat."
                )
            },
            {
                "id": "overview",
                "title": "University Overview, Accreditation, and Campus Location",
                "keywords": ["juit", "overview", "waknaghat", "solan", "campus", "acres", "naac", "nba", "nirf", "ranking", "established", "location", "address", "himachal"],
                "content": (
                    "Jaypee University of Information Technology (JUIT) was established in May 2002 under HP State Act No. 14 "
                    "as a public-private partnership between Jaiprakash Sewa Sansthan (JSS) and the Govt of Himachal Pradesh. "
                    "It is the first State Private University in Himachal Pradesh, located across 25 acres in Waknaghat, Kandaghat Tehsil, Solan District. "
                    "Accreditations include NAAC Grade A+ (CGPA 3.40), NBA programmatic accreditation for B.Tech Biotech, and NIRF 201-300 rank band. "
                    "It holds an FSSAI 'Eat Right Campus' certification and an A+ Grade from Green Rankings for sustainability."
                )
            },
            {
                "id": "leadership",
                "title": "University Administration, Leadership & State Governance",
                "keywords": ["chancellor", "vice chancellor", "vc", "cm", "chief minister", "sukhu", "governor", "pro chancellor", "chairman", "ceo", "registrar", "dean", "manoj gaur", "rajendra", "brigadier", "leadership", "administration"],
                "content": (
                    "University Administration & Leadership: "
                    "Vice Chancellor (VC) is Prof. (Dr.) Rajendra Kumar Sharma (Chief Academic & Operating Officer). "
                    "Chancellor is Sh. Kavinder Gupta (Hon'ble Governor of Himachal Pradesh). "
                    "Pro-Chancellor & Chairman is Sh. Manoj Gaur (Executive Chairman of Jaypee Group). "
                    "Chief Minister (CM) of Himachal Pradesh is Sukhvinder Singh Sukhu. "
                    "Chief Executive Officer (CEO) is Sh. Manu Bhaskar Gaur. "
                    "Registrar & Dean of Students is Brigadier RK Sharma, SM (Retd.). "
                    "Dean (Academics & Research) is Prof. (Dr.) Ashok Kumar Gupta. "
                    "Dean (Accreditations) and Controller of Examinations is Prof. (Dr.) Sunil Kumar Khah."
                )
            },
            {
                "id": "drobotics_ece",
                "title": "ECE Department and Drobotics Robotics Labs",
                "keywords": ["d-bot", "dbot", "drobot", "drobotics", "robot", "robotics", "lab", "ecl 5", "ecl 7", "ece", "electronics", "communication", "rajiv kumar", "eyantra", "embedded", "vlsi", "hardware"],
                "content": (
                    "Department of Electronics & Communication Engineering (ECE) is headed by Prof. (Dr.) Rajiv Kumar. "
                    "Key robotics and hardware laboratories include: "
                    "- ECL 5: Drobotics Lab / e-Yantra Robotics Lab (home of robotics development and D-Bot). "
                    "- ECL 7: Embedded Systems and Robotics Lab / Microcontroller Lab. "
                    "- ECL 1: Mitsubishi Factory Automation Lab with industrial PLC trainers. "
                    "- CL 1: VLSI and Signal Processing Lab. "
                    "- ECL 2: Digital Systems Design Lab, and ECL 4: Communication Systems Lab. "
                    "ECE offers B.Tech in ECE, B.Tech in Electronics & Computer Science (ECS), M.Tech in ECE, M.Tech in IoT, and Ph.D."
                )
            },
            {
                "id": "cse_it",
                "title": "Computer Science & Engineering and Information Technology (CSE/IT)",
                "keywords": ["cse", "it", "computer science", "information technology", "vivek sehgal", "software", "coding", "ai", "machine learning", "cyber security", "bca"],
                "content": (
                    "Department of CSE & IT is headed by Prof. (Dr.) Vivek Sehgal. "
                    "It offers B.Tech CSE, B.Tech IT, BCA (3-year), M.Tech CSE, and Ph.D. in CSE. "
                    "Specialization tracks for B.Tech include: Artificial Intelligence and Machine Learning (AI-ML), "
                    "Artificial Intelligence and Data Science (AI-DS), Cyber Security, UX-UI Design, and Full Stack Software Development. "
                    "Labs include AI Lab, Machine Learning Lab, Cloud Computing Lab, Data Science & Visualization Lab, Operating Systems Lab, and Networks Lab."
                )
            },
            {
                "id": "biotech",
                "title": "Biotechnology and Bioinformatics Department",
                "keywords": ["biotech", "bioinformatics", "biology", "sudhir kumar", "nba", "dna", "plant", "microbiology", "genomics", "culture"],
                "content": (
                    "Department of Biotechnology & Bioinformatics is headed by Prof. (Dr.) Sudhir Kumar. "
                    "Its B.Tech Biotechnology program holds prestigious NBA accreditation (2024–2027). "
                    "Offers B.Tech BT, B.Tech Bioinformatics, B.Sc., M.Sc. Biotechnology (DBT-supported), M.Sc. Microbiology, M.Tech BT, and Ph.D. "
                    "Facilities include Proteomics Lab, Genomics Lab, Plant Biotech & 160 sq.m Greenhouse, Cell Culture Lab, Bioreactor/Fermenter Lab, and High-End Instrumentation (DNA Sequencer, Q-ToF MS, HPLC)."
                )
            },
            {
                "id": "civil",
                "title": "Civil Engineering Department",
                "keywords": ["civil", "concrete", "structural", "surveying", "geotechnical", "hydraulics", "construction", "ashish kumar"],
                "content": (
                    "Department of Civil Engineering is headed by Prof. (Dr.) Ashish Kumar. "
                    "Offers B.Tech Civil, M.Tech (Structural Engineering, Construction Management, Environmental Engineering), and Ph.D. "
                    "Equipped with Concrete Technology Lab, Structural Mechanics Lab, Surveying Lab (total stations), Geotechnical Lab, Fluvial Hydraulics Lab, and industry tools like STAAD Pro, Revit, and OpenSees."
                )
            },
            {
                "id": "management_hss",
                "title": "Jaypee Business School / Humanities & Social Sciences (JBSW)",
                "keywords": ["bba", "mba", "business", "management", "amit srivastava", "hss", "marketing", "finance", "analytics", "hr"],
                "content": (
                    "Jaypee Business School Waknaghat (Dept of Humanities and Social Sciences) is headed by Prof. (Dr.) Amit Srivastava. "
                    "Offers BBA (3-year) with tracks in Business Analytics, Finance, Marketing, and Human Resource, plus MBA (2-year) and Ph.D. "
                    "Admission to BBA is based on CUET (UG) score or 10+2 qualifying marks (minimum 60%)."
                )
            },
            {
                "id": "math_physics",
                "title": "Departments of Mathematics, Physics & Materials Science",
                "keywords": ["math", "mathematics", "physics", "materials", "p.b. barman", "rakesh bajaj", "sunil khah", "optics", "semiconductor"],
                "content": (
                    "Department of Mathematics is headed by Prof. (Dr.) Rakesh Kumar Bajaj, providing core mathematics and computational analysis courses. "
                    "Department of Physics & Materials Science is headed by Prof. (Dr.) P.B. Barman and includes Dean Prof. (Dr.) Sunil Kumar Khah, "
                    "delivering engineering physics, optics, electromagnetics, and solid-state materials labs."
                )
            },
            {
                "id": "admissions",
                "title": "Admissions Process and Eligibility Criteria",
                "keywords": ["admission", "apply", "eligibility", "jee", "10+2", "cutoff", "seats", "application fee", "cuet", "gate"],
                "content": (
                    "Admissions for B.Tech are split equally: 50% seats via JEE Main All India Rank merit, and 50% via 10+2 board marks (Physics + Maths compulsory). "
                    "Application fee is Rs. 1,200 (online or DD). BBA admission requires 60% in 10+2 or CUET. "
                    "M.Tech admission is via GATE or institutional PGET. Ph.D. requires Master's degree (60% marks) or 4-year B.Tech (75%) plus entrance exam & interview."
                )
            },
            {
                "id": "fees_hostel",
                "title": "Tuition Fees, Hostel Charges, and Defense Rebates",
                "keywords": ["fee", "fees", "cost", "hostel", "mess", "food", "tuition", "rebate", "defense", "scholarship", "charges"],
                "content": (
                    "Fee Structure (per semester): "
                    "- B.Tech Tuition: Rs. 1,60,500 per semester. "
                    "- B.Tech (Defense & PMF wards): 30% rebate (Rs. 1,12,350 per semester). "
                    "- B.Tech (War Widows & PWD): 35% rebate (Rs. 1,04,325 per semester). "
                    "- BBA Tuition: Rs. 80,500 per semester. "
                    "- Ph.D. Tuition: Rs. 31,650 per semester. "
                    "- Hostel Charges: Rs. 80,000 per semester (including lodging, boarding, mess, laundry, and internet). "
                    "- One-time refundable caution money: Rs. 20,000; non-refundable admission fee: Rs. 25,000 (B.Tech) / Rs. 15,000 (BBA)."
                )
            },
            {
                "id": "campus_facilities",
                "title": "Campus Life, Hostels, Medical, and Student Facilities",
                "keywords": ["campus", "facilities", "hostel", "rooms", "gym", "sports", "mess", "food", "eat right", "medical", "doctor", "dispensary", "hospital", "wifi", "atm", "bank"],
                "content": (
                    "JUIT campus is a self-contained hill sanctuary with modern hostels equipped with Wi-Fi, heating, and laundry. "
                    "Food is served across spacious dining halls holding FSSAI 'Eat Right Campus' certification. "
                    "Healthcare: Dedicated Medical Dispensary with resident doctors, nursing staff, and 24/7 ambulance support. "
                    "Recreation: State-of-the-art gymnasiums, basketball courts, badminton courts, volleyball, cricket/football grounds, "
                    "and student activity centers. Banking facility with Punjab National Bank branch and ATM on campus."
                )
            }
        ]

        self.chunks = base_sections

        # Also ingest raw text if available for supplemental sentences
        if os.path.exists(self.data_path):
            try:
                with open(self.data_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                # Extract any additional paragraphs
                paragraphs = [p.strip() for p in content.split(".") if len(p.strip()) > 80]
                for i, p in enumerate(paragraphs[:30]):
                    tokens = [w.lower() for w in re.findall(r"\b[a-zA-Z]{3,}\b", p)]
                    if len(tokens) > 10:
                        self.chunks.append({
                            "id": f"raw_{i}",
                            "title": f"Institutional Record {i+1}",
                            "keywords": list(set(tokens))[:15],
                            "content": p + "."
                        })
            except Exception as e:
                pass

    def _build_index(self):
        # Precompute term frequencies for BM25-like scoring
        self.doc_freqs = Counter()
        self.doc_token_counts = []
        self.doc_tokens = []

        for chunk in self.chunks:
            tokens = self._tokenize(chunk["title"] + " " + " ".join(chunk["keywords"]) + " " + chunk["content"])
            self.doc_tokens.append(tokens)
            self.doc_token_counts.append(len(tokens))
            unique_tokens = set(tokens)
            for t in unique_tokens:
                self.doc_freqs[t] += 1

        self.num_docs = len(self.chunks)
        self.avg_doc_len = sum(self.doc_token_counts) / max(1, self.num_docs)

    def _tokenize(self, text: str) -> List[str]:
        return [w.lower() for w in re.findall(r"\b[a-zA-Z0-9_\-]{2,}\b", text)]

    def retrieve(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return self.chunks[:top_k]

        scores = []
        k1 = 1.5
        b = 0.75

        for idx, tokens in enumerate(self.doc_tokens):
            doc_len = self.doc_token_counts[idx]
            token_counts = Counter(tokens)
            score = 0.0

            # Boost if query matches keywords directly
            chunk_keywords = [k.lower() for k in self.chunks[idx].get("keywords", [])]

            for qt in query_tokens:
                if qt in chunk_keywords:
                    score += 4.0  # Big boost for direct keyword match

                if qt in self.doc_freqs:
                    df = self.doc_freqs[qt]
                    idf = math.log((self.num_docs - df + 0.5) / (df + 0.5) + 1.0)
                    tf = token_counts[qt]
                    denom = tf + k1 * (1 - b + b * (doc_len / self.avg_doc_len))
                    score += idf * (tf * (k1 + 1)) / denom

            scores.append((score, idx))

        scores.sort(key=lambda x: x[0], reverse=True)
        results = [self.chunks[idx] for score, idx in scores[:top_k] if score > 0]
        return results

    def format_context(self, query: str) -> str:
        relevant = self.retrieve(query, top_k=2)
        if not relevant:
            return ""
        lines = []
        for c in relevant:
            lines.append(f"[{c['title']}]: {c['content']}")
        return "\n".join(lines)


if __name__ == "__main__":
    engine = KnowledgeEngine()
    test_queries = [
        "Where is the Drobotics lab?",
        "Who is the Vice Chancellor of the university?",
        "What is the fee for BTech and hostel charges?",
        "Tell me about the CSE department and HOD."
    ]
    for q in test_queries:
        print(f"\nQuery: {q}")
        ctx = engine.format_context(q)
        print(f"Context: {ctx[:200]}...")
