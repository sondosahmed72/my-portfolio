// PortfolioPage.jsx
import React, { useState, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import profileImage from "./assets/images/profile.jpg"; // adjust the path based on your file location
import { FaGithub, FaLinkedin, FaEnvelope } from "react-icons/fa";

/**
 * Stand-alone portfolio page
 * - Tailwind CSS required
 * - framer-motion required
 *
 * Drop into your React app and render <PortfolioPage />.
 */

// ---------- Profile & GitHub ----------
const PROFILE_NAME = "Sondos Ahmed";
const LINKEDIN_URL = "https://www.linkedin.com/in/sondos-ahmed-109787246/";
const GITHUB = "https://github.com/sondosahmed72";
const EMAIL = "mailto:sondosahmed72@gmail.com";
const GITHUB_DISCOVER_URL = "https://github.com/sondosahmed72?tab=repositories";
const LINKEDIN_DISCOVER_URL =
  "https://www.linkedin.com/in/sondos-ahmed-109787246/details/certifications/";

// Example skill icons mapping

// ---------- YOUR DATA (exact as provided) ----------
const projectsData = [
  // ---------------- Graduation Projects ----------------
  {
    id: "ra7ala",
    category: "Graduation",
    title: "Ra7ala — Travel Planning & Itinerary Generator",
    year: "2025",
    tags: ["Graduation", "Full-Stack", "AI"],
    tech: "CSP, K-Means++, ACO, KD-Tree, DELG, FAISS, Angular, .NET Core, Flask",
    bullets: [
      "Built an intelligent itinerary generator using CSP, K-Means++, ACO, and KD-Tree for route optimization.",
      "Developed landmark recognition system with DELG + FAISS, achieving 92.38% accuracy.",
      "Integrated full-stack architecture: Angular (frontend), .NET Core (backend), Flask (AI services).",
    ],
    links: {
      github:
        "https://github.com/sondosahmed72/Ra7laa-Travel-Planning-Itinerary-Generator",
      live: "https://ra7ala.live",
    },
  },
  {
    id: "ssim",
    category: "Graduation",
    title: "SSIM — Site Safety Intelligent Monitor",
    year: "2025",
    tags: ["Graduation", "Computer Vision", "NTI"],
    tech: "PyTorch, YOLOv11, VLM, Streamlit",
    bullets: [
      "Built a real-time safety monitoring system using YOLOv11 for PPE detection and a Vision-Language Model for environmental hazard analysis.",
      "Integrated multi-threaded pipeline with async VLM tasks, image buffering, and violation tracking for persistent safety risks.",
      "Implemented Streamlit interface with live camera feed, logs, and alert system for automated notifications via Telegram.",
    ],
    links: {
      github:
        "https://github.com/sondosahmed72/SSIM-Site-Safety-Intelligence-System",
      live: "https://ra7ala.live",
    },
  },

  // ---------------- Machine Learning & AI ----------------
  {
    id: "waste_classification",
    category: "Machine Learning & AI",
    title: "Waste Classification Using Transfer Learning",
    year: "2025",
    tags: ["ML", "CNN", "Transfer Learning"],
    tech: "TensorFlow, Keras, VGG16, ImageDataGenerator, Python",
    bullets: [
      "Built a binary image classifier to distinguish between Organic (O) and Recyclable (R) waste.",
      "Applied transfer learning with VGG16, data augmentation, and fine-tuning to achieve high accuracy.",
      "Plotted training/validation accuracy and loss curves; visualized predictions on test images.",
    ],
    links: {
      github:
        "https://github.com/sondosahmed72/Waste-Classification-Using-Transfer-Learning",
      live: "https://ra7ala.live",
    },
  },
  {
    id: "cv_eval",
    category: "Machine Learning & AI",
    title: "CV Evaluation System",
    year: "2025",
    tags: ["AI", "LangGraph", "DeepSeek"],
    tech: "LangGraph, DeepSeek",
    bullets: [
      "Developed an automated CV screening system using LangGraph and DeepSeek for semantic analysis.",
      "Implemented multi-agent evaluation pipeline with skill-matching, scoring, and feedback generation.",
      "Produced HR-friendly structured JSON outputs for decision-making.",
    ],
    links: {
      github: "https://github.com/sondosahmed72/cv-evaluation-system",
      live: "https://ra7ala.live",
    },
  },
  {
    id: "adult_income",
    category: "Machine Learning & AI",
    title: "Adult Income Classification",
    year: "2025",
    tags: ["ML", "Classification"],
    tech: "scikit-learn, XGBoost, SMOTE",
    bullets: [
      "Performed EDA, data cleaning, encoding, and outlier treatment on UCI Adult Census dataset.",
      "Applied SMOTE and class-weighting to handle class imbalance; trained Logistic Regression, Random Forest, and XGBoost.",
      "Achieved ~88% accuracy using XGBoost with balanced minority-class performance.",
    ],
    links: {
      github: "https://github.com/sondosahmed72/Adult-Income-Classification",
      live: "https://ra7ala.live",
    },
  },
  {
    id: "boston_housing",
    category: "Machine Learning & AI",
    title: "Boston Housing Price Prediction",
    year: "2025",
    tags: ["ML", "Regression"],
    tech: "scikit-learn, Pandas, Seaborn",
    bullets: [
      "Performed extensive EDA and data cleaning including outlier treatment and normalization.",
      "Applied PCA for dimensionality reduction and variance interpretation.",
      "Trained Linear Regression and Random Forest Regressor achieving high R² and low RMSE.",
    ],
    links: {
      github:
        "https://github.com/sondosahmed72/Boston-Housing-Price-Prediction",
      live: "https://ra7ala.live",
    },
  },
  {
    id: "cirrhosis_pred",
    category: "Machine Learning & AI",
    title: "Cirrhosis Patient Survival Prediction",
    year: "2025",
    tags: ["ML", "Healthcare"],
    tech: "XGBoost, scikit-learn",
    bullets: [
      "Built a predictive model for cirrhosis survival outcomes using XGBoost.",
      "Implemented comprehensive preprocessing and stratified 5-fold cross-validation.",
      "Achieved 86.06% accuracy and LogLoss: 0.369 on validation.",
    ],
    links: {
      github:
        "https://github.com/sondosahmed72/Cirrhosis-Patient-Survival-Prediction-XGBoost-Classifier",
      live: "https://ra7ala.live",
    },
  },
  {
    id: "credit_score",
    category: "Machine Learning & AI",
    title: "Credit Scoring Model",
    year: "2024",
    tags: ["ML", "Finance"],
    tech: "scikit-learn",
    bullets: [
      "Developed a creditworthiness evaluation model using classification techniques.",
    ],
    links: {
      github:
        "https://github.com/sondosahmed72/CodeAlpha-ML-Internship-Projects/tree/main/Credit%20Scoring%20Model",
      live: "https://ra7ala.live",
    },
  },
  {
    id: "income_analysis",
    category: "Machine Learning & AI",
    title: "Income Analysis",
    year: "2023",
    tags: ["ML", "EDA"],
    tech: "Python, ML",
    bullets: [
      "Explored demographic income data and built a predictive income classification model.",
    ],
    links: {
      github: "https://github.com/sondosahmed72/Adult-Income-Classification",
      live: "https://ra7ala.live",
    },
  },
  {
    id: "breast_cancer",
    category: "Machine Learning & AI",
    title: "Breast Cancer Classification",
    year: "2024",
    tags: ["ML", "Healthcare"],
    tech: "Python, scikit-learn",
    bullets: [
      "Built binary classifier to distinguish malignant vs benign tumors using multiple models.",
      "Compared Logistic Regression, Decision Trees, and Random Forest for best accuracy.",
    ],
    links: {
      github:
        "https://github.com/sondosahmed72/CognoRise-InfoTech/tree/main/task1",
      live: "https://ra7ala.live",
    },
  },

  // ---------------- Deep Learning & Computer Vision ----------------
  {
    id: "agri_land_classification",
  category: "Deep Learning & Computer Vision",
  title: "Agricultural Land Classification using CNNs & Vision Transformers",
  year: "2025",
  tags: ["DL", "CNN", "Vision Transformer", "Geospatial AI"],
  tech: "Keras, PyTorch, CNNs, Vision Transformers, Data Augmentation, Transfer Learning",
  bullets: [
    "Built and trained CNN and vision transformer models for classifying agricultural land using satellite imagery.",
    "Implemented efficient data loading and augmentation pipelines using Keras and PyTorch, including a custom geospatial data loader.",
    "Applied transfer learning to fine-tune pre-trained models for improved classification accuracy.",
    "Evaluated model performance using accuracy, F1-score, and AU-ROC metrics and conducted comparative analysis.",
    "Documented methodology, results, and insights in a professional project report."
  ],
  links: {
    github: "https://github.com/sondosahmed72/Agricultural-Land-Classification-using-CNNs-Vision-Transformers.git", // ضع رابط الريبو لو متاح
    live: ""    // ضع رابط المشروع لو متاح على الويب
  },
  },
  {
    id: "neural_style",
    category: "Deep Learning & Computer Vision",
    title: "Neural Style Transfer",
    year: "2025",
    tags: ["DL", "PyTorch"],
    tech: "PyTorch, VGG19",
    bullets: [
      "Implemented Neural Style Transfer to merge the style of one image with the content of another.",
      "Used pretrained VGG19 to compute content and style losses.",
    ],
    links: {
      github:
        "https://github.com/sondosahmed72/Neural-Style-Transfer-with-Pytorch",
      live: "https://ra7ala.live",
    },
  },
  {
    id: "srcnn",
    category: "Deep Learning & Computer Vision",
    title: "Image Super-Resolution using CNN (SRCNN)",
    year: "2025",
    tags: ["DL", "CV"],
    tech: "PyTorch, SRCNN",
    bullets: [
      "Developed improved SRCNN architecture with deep convolutional layers and batch normalization.",
      "Evaluated performance with SSIM, PSNR, and MSE metrics, achieving superior visual quality.",
    ],
    links: {
      github:
        "https://github.com/sondosahmed72/Super-Resolution-Image-Enhancement-using-CNN-SRCNN-",
      live: "https://ra7ala.live",
    },
  },
  {
    id: "anime_vae",
    category: "Deep Learning & Computer Vision",
    title: "Anime Face Generation using VAE",
    year: "2025",
    tags: ["DL", "VAE"],
    tech: "TensorFlow, Keras",
    bullets: [
      "Built Variational Autoencoder from scratch to generate realistic anime faces.",
      "Implemented reconstruction and KL divergence losses for training.",
    ],
    links: {
      github:
        "https://github.com/sondosahmed72/Anime-Face-Generation-using-Variational-Autoencoder-VAE-from-Scratch",
      live: "https://ra7ala.live",
    },
  },
  {
    id: "rag_video",
    category: "Deep Learning & Computer Vision",
    title: "RAG with Video — Video Q&A & Summarization",
    year: "2025",
    tags: ["DL", "Multimodal"],
    tech: "Transformers, BLIP, Whisper, Sentence-Transformers",
    bullets: [
      "Built Retrieval-Augmented Generation (RAG) system combining transcript and visual embeddings.",
      "Integrated Whisper, BLIP, and Sentence-Transformers; deployed GUI with Gradio.",
    ],
    links: {
      github: "https://github.com/sondosahmed72/rag-with-video",
      live: "https://ra7ala.live",
    },
  },
  {
    id: "car_ocr",
    category: "Deep Learning & Computer Vision",
    title: "Car Detection and License Plate OCR",
    year: "2025",
    tags: ["CV", "OCR"],
    tech: "YOLOv8, PaddleOCR, OpenCV",
    bullets: [
      "Trained YOLOv8 on annotated dataset and implemented OCR for Egyptian plates.",
      "Built complete pipeline for detection, cropping, and text recognition.",
    ],
    links: {
      github:
        "https://github.com/sondosahmed72/Car-Detection-and-License-Plate-OCR-YOLOv8-PaddleOCR-",
      live: "https://ra7ala.live",
    },
  },
  {
    id: "fer",
    category: "Deep Learning & Computer Vision",
    title: "Facial Expression Recognition (FER-2013)",
    year: "2025",
    tags: ["CV", "Emotion"],
    tech: "PyTorch, CNN",
    bullets: [
      "Trained CNN to classify 7 emotions on FER-2013 dataset using data augmentation and dropout.",
    ],
    links: {
      github:
        "https://github.com/sondosahmed72/Facial-Emotion-Classification-PyTorch",
      live: "https://ra7ala.live",
    },
  },
  {
    id: "dog_cat",
    category: "Deep Learning & Computer Vision",
    title: "Dog vs Cat Classification",
    year: "2024",
    tags: ["CNN", "Transfer Learning"],
    tech: "TensorFlow, InceptionV3",
    bullets: [
      "Developed image classifiers using custom CNN and transfer learning with InceptionV3.",
    ],
    links: {
      github: "https://github.com/sondosahmed72/Dog-vs-Cat-Classification-",
      live: "https://ra7ala.live",
    },
  },
  {
    id: "img_class",
    category: "Deep Learning & Computer Vision",
    title: "Image Classification",
    year: "2024",
    tags: ["CNN", "TensorFlow"],
    tech: "TensorFlow, CNN",
    bullets: [
      "Developed CNN models for image classification and applied data augmentation for better accuracy.",
    ],
    links: {
      github:
        "https://github.com/sondosahmed72/-Neural-Network-Image-Classification-with-TensorFlow",
      live: "https://ra7ala.live",
    },
  },
  {
    id: "mnist",
    category: "Deep Learning & Computer Vision",
    title: "Handwritten Digit Recognition (MNIST)",
    year: "2024",
    tags: ["CNN"],
    tech: "TensorFlow",
    bullets: [
      "Trained CNN on MNIST dataset achieving high accuracy in digit classification.",
    ],
    links: {
      github:
        "https://github.com/sondosahmed72/CNN-based-Handwritten-Digit-Recognition-using-MNIST-Dataset",
      live: "https://ra7ala.live",
    },
  },

  // ---------------- NLP ----------------
  {
    id: "spotify_rec",
    category: "NLP",
    title: "Spotify Recommendation System",
    year: "2024",
    tags: ["NLP", "Recommender"],
    tech: "NLP, Python",
    bullets: [
      "Built collaborative and content-based filtering systems to suggest music.",
      "Used cosine similarity and user profiling for personalized recommendations.",
    ],
    links: {
      github:
        "https://github.com/sondosahmed72/Spotify-s-recommendation-system-",
      live: "https://ra7ala.live",
    },
  },
  {
    id: "twitter_sentiment",
    category: "NLP",
    title: "Twitter Sentiment Analysis",
    year: "2024",
    tags: ["NLP", "Sentiment Analysis"],
    tech: "scikit-learn, NLP",
    bullets: [
      "Processed and analyzed 160k+ tweets using text preprocessing and classification pipelines.",
      "Achieved high accuracy using TF-IDF with SVM and Logistic Regression.",
    ],
    links: {
      github:
        "https://github.com/sondosahmed72/CodeAlpha-ML-Internship-Projects/tree/main/TWITTER%20SENTIMENT%20ANALYSIS",
      live: "https://ra7ala.live",
    },
  },
  {
    id: "restaurant_reviews",
    category: "NLP",
    title: "Restaurant Review Classification",
    year: "2024",
    tags: ["NLP", "Text Classification"],
    tech: "Python, NLP",
    bullets: [
      "Classified customer reviews using lemmatization, tokenization, and ML classifiers.",
    ],
    links: {
      github: "https://github.com/sondosahmed72/NLP-task",
      live: "https://ra7ala.live",
    },
  },

  // ---------------- Data Analysis & BI ----------------
  {
    id: "stock_dashboard",
    category: "Data Analysis & BI",
    title: "Analyzing Stock & Revenue Data Dashboard",
    year: "2025",
    tags: ["Data Analysis", "Dashboard"],
    tech: "BeautifulSoup, yfinance, Plotly",
    bullets: [
      "Scraped and merged stock & revenue data, visualized trends with interactive Plotly charts.",
      "Developed a Jupyter dashboard to monitor company performance over time.",
    ],
    links: {
      github: "https://github.com/sondosahmed72/Tesla-GameStop-Dashboard",
      live: "https://ra7ala.live",
    },
  },
  {
    id: "titanic",
    category: "Data Analysis & BI",
    title: "Titanic Dataset Analysis & EDA",
    year: "2025",
    tags: ["EDA", "Python"],
    tech: "Pandas, NumPy, Seaborn",
    bullets: [
      "Cleaned data, engineered features, and visualized survival insights using Seaborn and Matplotlib.",
    ],
    links: {
      github: "https://github.com/sondosahmed72/Titanic-Dataset-Analysis-EDA",
      live: "https://ra7ala.live",
    },
  },
  {
    id: "telco_churn",
    category: "Data Analysis & BI",
    title: "Telco Customer Churn EDA",
    year: "2025",
    tags: ["EDA", "Python"],
    tech: "Pandas, Seaborn",
    bullets: [
      "Performed cleaning, feature engineering, and churn pattern analysis using visualization.",
    ],
    links: {
      github: "https://github.com/sondosahmed72/Telco-Customer-Churn-EDA",
      live: "https://ra7ala.live",
    },
  },
  {
    id: "flight_price",
    category: "Data Analysis & BI",
    title: "Flight Price Prediction",
    year: "2024",
    tags: ["Regression", "EDA"],
    tech: "Python, ML",
    bullets: [
      "Scraped flight booking data and trained regression model for price forecasting.",
    ],
    links: {
      github: "https://github.com/sondosahmed72/Flight-Price-Prediction",
      live: "https://ra7ala.live",
    },
  },
  {
    id: "market_sales",
    category: "Data Analysis & BI",
    title: "Market Sales Analysis",
    year: "2024",
    tags: ["Power BI"],
    tech: "Power BI",
    bullets: [
      "Built interactive dashboards to visualize and analyze market sales trends.",
    ],
    links: {
      github:
        "https://github.com/sondosahmed72/Market-Sales-analysis-by-Power-Bi",
      live: "https://ra7ala.live",
    },
  },
  {
    id: "kiva_loans",
    category: "Data Analysis & BI",
    title: "Kiva Loans Data Analysis",
    year: "2024",
    tags: ["Power BI", "Python"],
    tech: "Python, Power BI",
    bullets: [
      "Analyzed micro-loan data to extract global lending insights using Power BI.",
    ],
    links: {
      github: "https://github.com/sondosahmed72/Kiva-Loans-Analysis",
      live: "https://ra7ala.live",
    },
  },
  {
    id: "student_perf",
    category: "Data Analysis & BI",
    title: "Student Performance Analysis",
    year: "2023",
    tags: ["R", "Statistics"],
    tech: "R",
    bullets: [
      "Conducted statistical analysis to identify key factors influencing student grades.",
    ],
    links: {
      github: "https://github.com/sondosahmed72/Student-Performance-",
      live: "https://ra7ala.live",
    },
  },
];

const projectsCategories = [
  "All",
  "Graduation",
  "Machine Learning & AI",
  "Deep Learning & Computer Vision",
  "NLP",
  "Data Analysis & BI",
];

const certCategories = [
  "All",
  "AI & Machine Learning",
  "Deep Learning & Computer Vision",
  "NLP & LLM",
  "Data Analysis & BI",
  "Data Engineering & Databases",
  "Programming & Problem Solving",
  "Tools & Cloud",
  "Mobile & Frontend",
  "Software Testing",
];

const certificationsData = [
  // Deep Learning & Computer Vision
  {
    id: "dl_cv_coursera",
    category: "Deep Learning & Computer Vision",
    title:
      "Deep Learning for Computer Vision — University of Colorado Boulder | Coursera",
    bullets: [
      "Improved model performance and training stability using MLPs and normalization techniques.",
      "Implemented autoencoders for unsupervised feature learning and designed GANs for synthetic image generation.",
      "Trained CNNs for image classification tasks and understood spatial feature extraction from visual data.",
      "Applied advanced architectures like ResNet for deep image recognition and U-Net for image segmentation.",
    ],
  },
  {
    id: "dl2_pytorch",
    category: "Deep Learning & Computer Vision",
    title: "Deep Learning with Keras & TensorFlow — IBM | Nov 2025",
    bullets: [
      "Created custom layers and models in Keras and integrated with TensorFlow 2.x.",
      "Developed advanced CNNs for image tasks and Transformer models for sequential data.",
      "Studied unsupervised learning, Deep Q-networks (DQNs), and reinforcement learning.",
      "Gained skills in Deep Learning, Neural Networks, NLP, Generative AI, Image Analysis, and Performance Tuning.",
    ],
  },
  {
    id: "dl_pytorch",
    category: "Deep Learning & Computer Vision",
    title: "Deep Learning with PyTorch — IBM | Nov 2025",
    bullets: [
      "Built and trained deep and convolutional neural networks using PyTorch.",
    ],
  },
  {
    id: "intro_keras",
    category: "Deep Learning & Computer Vision",
    title:
      "Introduction to Deep Learning & Neural Networks with Keras — IBM | Nov 2025",
    bullets: [
      "Designed and trained CNNs, RNNs, Transformers, and autoencoders.",
    ],
  },
  {
    id: "nvidia_dl",
    category: "Deep Learning & Computer Vision",
    title: "Getting Started with Deep Learning — NVIDIA | Sep 2025",
    bullets: [
      "Focused on intelligent system design and predictive models using AI, ML, and DL.",
    ],
  },
  {
    id: "biomedical_img",
    category: "Deep Learning & Computer Vision",
    title: "Biomedical Image Analysis in Python — DataCamp | Oct 2024",
    bullets: ["Analyzed biomedical images using Python, OpenCV, and Keras."],
  },
  {
    id: "img_modeling_keras",
    category: "Deep Learning & Computer Vision",
    title: "Image Modeling with Keras — DataCamp | Oct 2024",
    bullets: ["Developed image models using Keras."],
  },
  {
    id: "img_processing",
    category: "Deep Learning & Computer Vision",
    title: "Image Processing in Python — DataCamp | Oct 2024",
    bullets: [
      "Image transformations, augmentation, and CNN modeling using OpenCV and Keras.",
    ],
  },

  // AI & Machine Learning
  {
    id: "intro_pytorch",
    category: "AI & Machine Learning",
    title: "Introduction to Neural Networks with PyTorch — IBM | Oct 2025",
    bullets: [
      "Implemented linear/logistic regression and PyTorch tensor operations.",
    ],
  },
  {
    id: "ml_python_ibm",
    category: "AI & Machine Learning",
    title: "Machine Learning with Python — IBM | Sep 2025",
    bullets: [
      "Supervised/unsupervised models, regression, dimensionality reduction, capstone project.",
    ],
  },
  {
    id: "dl_specialization",
    category: "Deep Learning & Computer Vision",
    title: "Deep Learning Specialization — DeepLearning.AI | Sep 2025",
    bullets: [
      "Sequence models, neural networks, hyperparameter tuning, regularization, and optimization.",
    ],
  },
  {
    id: "ml_specialization",
    category: "AI & Machine Learning",
    title:
      "Machine Learning Specialization — Stanford & DeepLearning.AI | Aug 2024",
    bullets: [
      "Supervised learning, regression/classification, unsupervised methods, and advanced ML techniques.",
    ],
  },
  {
    id: "ts_ml_datacamp",
    category: "AI & Machine Learning",
    title: "Machine Learning for Time Series Data — DataCamp | Sep 2024",
    bullets: ["Time-series feature engineering and forecasting techniques."],
  },
  {
    id: "aws_ml_essentials",
    category: "AI & Machine Learning",
    title:
      "Machine Learning Essentials for Business & Technical Decision Makers — AWS | Jun 2024",
    bullets: ["ML concepts and business applications using Python."],
  },
  {
    id: "aws_ml_possible",
    category: "AI & Machine Learning",
    title:
      "Introduction to Machine Learning: Art of the Possible — AWS | Mar 2024",
    bullets: ["Overview of ML applications and Python implementation."],
  },
  {
    id: "aws_ml_terminology",
    category: "AI & Machine Learning",
    title: "Machine Learning Terminology and Process — AWS | Mar 2024",
    bullets: ["ML process, terminology, and workflow in Python."],
  },
  {
    id: "aws_ml_project",
    category: "AI & Machine Learning",
    title: "Planning a Machine Learning Project — AWS | Mar 2024",
    bullets: ["Project planning and implementation steps for ML."],
  },
  {
    id: "ds_diploma",
    category: "AI & Machine Learning",
    title:
      "Data Science and Machine Learning Diploma — CLS Learning Solutions | Apr 2024",
    bullets: [
      "Predictive modeling, ML techniques, time series, web scraping, BI, Flask, Python, and neural networks.",
    ],
  },

  // NLP & LLM
  {
    id: "text_embeddings",
    category: "NLP & LLM",
    title:
      "Understanding and Applying Text Embeddings — DeepLearning.AI | Aug 2024",
    bullets: ["Embedding techniques for semantic search, LLMs, and NLP tasks."],
  },
  {
    id: "hf_datacamp",
    category: "NLP & LLM",
    title: "Working with Hugging Face — DataCamp | Sep 2025",
    bullets: [
      "Hands-on experience with Hugging Face Transformers and Datasets.",
    ],
  },
  {
    id: "langchain_func",
    category: "NLP & LLM",
    title:
      "Functions, Tools, and Agents with LangChain — DeepLearning.AI | Sep 2025",
    bullets: ["LangChain usage for building LLM-based applications."],
  },
  {
    id: "langchain_app",
    category: "NLP & LLM",
    title:
      "LangChain for LLM Application Development — DeepLearning.AI | Sep 2025",
    bullets: ["Developed LLM-powered applications with LangChain framework."],
  },

  // Data Analysis & BI
  {
    id: "plotly_dash",
    category: "Data Analysis & BI",
    title: "Data Visualization with Plotly & Dash — DataCamp | Jul 2025",
    bullets: ["Interactive dashboards using Plotly and Dash."],
  },
  {
    id: "improve_data_viz",
    category: "Data Analysis & BI",
    title: "Improving Your Data Visualizations in Python — DataCamp | Aug 2024",
    bullets: ["Best practices for clear and effective graphs."],
  },
  {
    id: "power_bi",
    category: "Data Analysis & BI",
    title: "Introduction to Power BI — DataCamp | Aug 2024",
    bullets: ["Building dashboards and reporting with Power BI."],
  },
  {
    id: "ms_powerbi_path",
    category: "Data Analysis & BI",
    title:
      "Discover Data Analysis / Power BI Path — Microsoft Learning | Jul 2024",
    bullets: ["Microsoft learning path for data analysis and Power BI."],
  },
  {
    id: "get_data_powerbi",
    category: "Data Analysis & BI",
    title: "Get Data with Power BI Desktop — Microsoft | Jul 2024",
    bullets: ["Data cleaning and preparation in Power BI."],
  },
  {
    id: "get_started_powerbi",
    category: "Data Analysis & BI",
    title: "Get Started Building with Power BI — Microsoft | Jul 2024",
    bullets: ["Hands-on practice with Power BI dashboards."],
  },
  {
    id: "digital_transformation_data",
    category: "Data Analysis & BI",
    title: "Digital Transformation and Data Literacy — CIB Egypt | Aug 2024",
    bullets: ["Data literacy and digital transformation knowledge."],
  },

  // Data Engineering & Databases
  {
    id: "data_eng_ibm",
    category: "Data Engineering & Databases",
    title: "Introduction to Data Engineering — IBM | Sep 2025",
    bullets: ["Relational Databases, NoSQL, and Big Data Engines."],
  },
  {
    id: "databases_sql",
    category: "Data Engineering & Databases",
    title: "Databases and SQL for Data Science — IBM | Sep 2024",
    bullets: ["SQL for data extraction and transformation."],
  },
  {
    id: "data1",
    category: "Data Engineering & Databases",
    title: "Introduction to Data Literacy — DataCamp | 2024",
    bullets: ["Importance of data literacy and best practices."],
  },
  {
    id: "data2",
    category: "Data Engineering & Databases",
    title: "Introduction to Data — DataCamp | 2024",
    bullets: ["Data concepts and smart decision-making principles."],
  },
  {
    id: "data3",
    category: "Data Engineering & Databases",
    title: "Data Literacy Essentials / In Practice — SAS | 2024",
    bullets: [
      "Prepare, analyze, and visualize data using real-world examples.",
    ],
  },

  // Programming & Problem Solving
  {
    id: "python_meta",
    category: "Programming & Problem Solving",
    title: "Programming in Python — Meta / Coursera | Aug 2023",
    bullets: ["Core Python programming and best practices."],
  },
  {
    id: "intro_cs",
    category: "Programming & Problem Solving",
    title: "Introduction to Computer Science — Developer Career | Dec 2022",
    bullets: ["Foundations of computer science."],
  },
  {
    id: "problem_solving",
    category: "Programming & Problem Solving",
    title: "Problem Solving Level 1 — Coach Academy | 2022",
    bullets: ["Progressive problem solving training."],
  },
  {
    id: "problem_solving1",
    category: "Programming & Problem Solving",
    title: "Problem Solving Level 2 — Coach Academy | 2024",
    bullets: ["Advanced problem solving exercises."],
  },
  {
    id: "back_end_meta",
    category: "Programming & Problem Solving",
    title: "Back-End Developer — Meta / Coursera | Apr 2023",
    bullets: [
      "Back-end development concepts, APIs, databases, and server-side Python.",
    ],
  },
  {
    id: "flutter_training",
    category: "Mobile & Frontend",
    title: "Flutter Summer Training — Support ASU | Sep 2023",
    bullets: ["Dart, APIs, SQLite, Firebase, and UI development with Flutter."],
  },

  // Tools & Cloud
  {
    id: "sagemaker",
    category: "Tools & Cloud",
    title: "Introduction to Amazon SageMaker — AWS | Jun 2024",
    bullets: ["Basics of SageMaker for training and deploying ML models."],
  },
  {
    id: "aws_concepts",
    category: "Tools & Cloud",
    title: "AWS Concepts — DataCamp | Aug 2025",
    bullets: ["Foundational AWS knowledge for ML and cloud services."],
  },
  {
    id: "git_github",
    category: "Tools & Cloud",
    title: "Git and GitHub — Almdrasa | Sep 2023",
    bullets: ["Version control and collaboration."],
  },
  {
    id: "version_control",
    category: "Tools & Cloud",
    title: "Version Control — Meta / Coursera | Sep 2023",
    bullets: ["Fundamentals of Git workflows."],
  },

  // Software Testing
  {
    id: "software_testing",
    category: "Software Testing",
    title: "Software Testing Workshop — Support ASU | Feb 2023",
    bullets: ["ISTQB testing fundamentals and practices."],
  },
];

const skills = {
  programming: [
    "Python",
    "R",
    "C++",
    "Java",
    "C#",
    "JavaScript",
    "Dart",
    "SQL",
  ],
  AI: [
    "Computer Vision",
    "OCR",
    "NLP",
    "Feature Engineering",
    "Model Optimization",
    "Fine-Tuning",
    "Time-Series",
    "Transforms",
    "AutoEncoders",
    "LLMs",
    "LLM Agents",
    "Llama",
  ],
  Frameworks: [
    "PyTorch",
    "TensorFlow",
    "Keras",
    "scikit-learn",
    "Hugging Face",
    "OpenCV",
    "FAISS",
  ],
  Data: ["ETL", "Data Warehousing", "Apache Spark", "Hadoop", "Airflow"],
  vizualization: ["Power BI", "Matplotlib", "Seaborn", "Plotly"],
  DB: ["SQL Server", "MySQL", "Oracle", "MongoDB", "NoSQL"],
  Deployment: [
    "React",
    "Angular",
    "Flutter",
    "Flask",
    "Node.js",
    "ASP.NET MVC",
    "Docker",
  ],
  Tools: ["Git", "Jupyter Notebook", "SSIS", "Google Colab", "VS Code"],
};

const experiences = [
  {
    id: "nti_intern",
    title: "Internship Trainee — Advanced AI",
    org: "National Telecommunication Institute (NTI)",
    period: "Jun 2025 – Oct 2025",
    bullets: [
      "Explored advanced AI theory: optimizers (SGD, Adam, RMSProp), Batch Normalization, and Regularization.",
      "Developed computer vision projects for detection and segmentation using PyTorch, OpenCV, ResNet, EfficientNet, YOLO, and SAM.",
      "Built NLP pipelines for Arabic and English using transformers and embedding-based models.",
      "Designed AI agents with LangChain, LangGraph, and Pydantic.",
      "Managed data pipelines and warehousing using MS SQL Server, SSIS, and created BI dashboards in Power BI.",
    ],
  },
];

// ---------- Component ----------
export default function PortfolioPage() {
  const [projTag, setProjTag] = useState("All");
  const [projSearch, setProjSearch] = useState("");

  const [certTag, setCertTag] = useState("All");
  const [certSearch, setCertSearch] = useState("");

  const [skillSearch, setSkillSearch] = useState("");
  const filteredProjects = useMemo(() => {
    return projectsData.filter((p) => {
      if (projTag !== "All" && p.category !== projTag) return false;
      if (
        projSearch &&
        !`${p.title} ${p.tech} ${p.bullets.join(" ")}`
          .toLowerCase()
          .includes(projSearch.toLowerCase())
      )
        return false;
      return true;
    });
  }, [projTag, projSearch]);

  const filteredCerts = useMemo(() => {
    return certificationsData
      ? certificationsData.filter((c) => {
          if (certTag !== "All" && c.category !== certTag) return false;
          if (
            certSearch &&
            !`${c.title} ${c.bullets.join(" ")}`
              .toLowerCase()
              .includes(certSearch.toLowerCase())
          )
            return false;
          return true;
        })
      : [];
  }, [certTag, certSearch]);

  return (
    <div className="min-h-screen relative bg-slate-950 text-slate-100 overflow-hidden">
      {/* Animated Background */}
      <div className="fixed inset-0 z-0">
        {/* Base gradient */}
        <div className="absolute inset-0 bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900"></div>

        {/* Animated gradient orbs */}
        <div className="absolute top-0 -left-4 w-96 h-96 bg-violet-500/30 rounded-full mix-blend-multiply filter blur-3xl opacity-70 animate-blob"></div>
        <div className="absolute top-0 -right-4 w-96 h-96 bg-indigo-500/30 rounded-full mix-blend-multiply filter blur-3xl opacity-70 animate-blob animation-delay-2000"></div>
        <div className="absolute -bottom-8 left-20 w-96 h-96 bg-purple-500/30 rounded-full mix-blend-multiply filter blur-3xl opacity-70 animate-blob animation-delay-4000"></div>

        {/* Grid pattern overlay */}
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#80808012_1px,transparent_1px),linear-gradient(to_bottom,#80808012_1px,transparent_1px)] bg-[size:64px_64px]"></div>

        {/* Radial gradient overlay for depth */}
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,transparent_0%,rgba(15,23,42,0.8)_100%)]"></div>
      </div>

      <div className="relative z-10  mx-auto p-4">
        <style jsx>{`
          @keyframes blob {
            0%,
            100% {
              transform: translate(0px, 0px) scale(1);
            }
            33% {
              transform: translate(30px, -50px) scale(1.1);
            }
            66% {
              transform: translate(-20px, 20px) scale(0.9);
            }
          }
          .animate-blob {
            animation: blob 7s infinite;
          }
          .animation-delay-2000 {
            animation-delay: 2s;
          }
          .animation-delay-4000 {
            animation-delay: 4s;
          }
        `}</style>
        <div className="mx-auto p-8">
          {/* Header */}
          <header className="relative z-10 flex items-center justify-between gap-4">
            <div className="flex items-center gap-6">
              <div>
                <h1 className="text-3xl md:text-5xl font-bold tracking-tight text-white">
                  {PROFILE_NAME}
                </h1>
                <p className="mt-2 text-lg md:text-xl text-violet-200/70">
                  AI Engineer — Junior · FCIS Ain-Shams '25
                </p>
              </div>
            </div>

            <nav className="flex items-center gap-3 flex-wrap">
              {/* Section Links */}
              <a
                href="#projects"
                className="px-3 py-2 rounded-md text-sm bg-violet-500/20 ring-1 ring-violet-400/30 hover:bg-violet-500/30 transition"
              >
                Projects
              </a>
              <a
                href="#certs"
                className="px-3 py-2 rounded-md text-sm bg-violet-500/15 ring-1 ring-violet-400/25 hover:bg-violet-500/25 transition"
              >
                Certifications
              </a>

              {/* External Links */}
              <a
                href={GITHUB}
                target="_blank"
                rel="noreferrer"
                className="hover:text-violet-400 transition"
              >
                <FaGithub />
              </a>
              <a
                href={LINKEDIN_URL}
                target="_blank"
                rel="noreferrer"
                className="hover:text-violet-400 transition"
              >
                <FaLinkedin />
              </a>
              <a href={EMAIL} className="hover:text-violet-400 transition">
                <FaEnvelope />
              </a>
            </nav>
          </header>

          {/* Hero */}
          <section className="mt-8 grid md:grid-cols-3 gap-6 items-center">
            <div className="md:col-span-2 bg-slate-800/40 backdrop-blur-sm border border-slate-700/50 rounded-2xl p-6 shadow-xl">
              <motion.h2
                initial={{ x: -12, opacity: 0 }}
                animate={{ x: 0, opacity: 1 }}
                transition={{ type: "spring", stiffness: 90 }}
                className="text-2xl md:text-3xl font-bold text-white drop-shadow-lg"
              >
                Junior AI Engineer
              </motion.h2>
              <p className="mt-3 text-slate-200 leading-relaxed">
                Experienced in building end-to-end AI pipelines for computer
                vision, NLP, and full-stack AI products. Proficient in PyTorch,
                TensorFlow, and scikit-learn.
              </p>

              <div className="mt-5 flex flex-wrap gap-3">
                <a
                  href={
                    "https://drive.google.com/file/d/1ycQL0RIIDDH26OeSaQiTyh-mQNnLokoj/view?usp=sharing"
                  }
                  className="px-4 py-2 bg-violet-600 hover:bg-violet-500 rounded-full text-sm font-medium text-white shadow-lg hover:shadow-violet-500/50 hover:scale-105 transition-all"
                >
                  Download CV
                </a>
              </div>
            </div>

            {/* Profile Image */}
            <div className="relative w-64 h-64 rounded-full overflow-hidden ring-4 ring-violet-500/40 shadow-2xl shadow-violet-500/20 mx-auto">
              <img
                src={profileImage}
                alt="Sondos Ahmed"
                className="w-full h-full object-cover"
              />
            </div>
          </section>

          {/* Experience */}
          <section className="mt-8 w-full">
            <div className="bg-slate-800/40 backdrop-blur-sm p-6 rounded-2xl border border-slate-700/50 shadow-lg w-full">
              <h3 className="text-sm font-semibold text-violet-300 uppercase tracking-wider">
                Experience
              </h3>
              <div className="mt-4 flex flex-col gap-4">
                {experiences.map((e) => (
                  <div
                    key={e.id}
                    className="inline-block p-4 rounded-lg bg-slate-700/30 border border-slate-600/40 max-w-fit hover:bg-slate-700/40 transition"
                  >
                    <div className="flex items-center justify-between">
                      <div>
                        <div className="font-medium text-white">{e.title}</div>
                        <div className="text-xs text-violet-400">
                          {e.org} | {e.period}
                        </div>
                      </div>
                    </div>
                    <ul className="mt-2 text-sm text-slate-300 list-disc ml-5 space-y-1">
                      {e.bullets.map((b, i) => (
                        <li key={i}>{b}</li>
                      ))}
                    </ul>
                  </div>
                ))}
              </div>
            </div>
          </section>

          {/* Skills */}
          <section className="mt-12 w-full">
            <div className="w-full p-6 bg-slate-800/40 backdrop-blur-sm rounded-2xl border border-slate-700/50 shadow-lg">
              <h3 className="text-xl font-semibold text-violet-300 mb-4 uppercase tracking-wider">
                Key Skills
              </h3>

              <input
                type="text"
                placeholder="Search skills..."
                value={skillSearch}
                onChange={(e) => setSkillSearch(e.target.value)}
                className="w-full mb-6 px-3 py-2 rounded-md bg-slate-700/30 text-white placeholder:text-slate-400 border border-slate-600/40 focus:outline-none focus:ring-2 focus:ring-violet-500/50"
              />

              <div className="grid sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                {Object.entries(skills).map(([category, arr]) => (
                  <div
                    key={category}
                    className="bg-slate-700/30 p-4 rounded-2xl border border-slate-600/40 shadow-sm hover:bg-slate-700/40 transition"
                  >
                    <h4 className="text-sm font-semibold text-violet-300 mb-3 capitalize uppercase tracking-wide">
                      {category}
                    </h4>
                    <div className="flex flex-wrap gap-2">
                      {arr
                        .filter((s) =>
                          s.toLowerCase().includes(skillSearch.toLowerCase())
                        )
                        .map((s) => (
                          <span
                            key={s}
                            className="flex items-center gap-2 text-xs px-2 py-1 rounded-md bg-violet-500/20 text-slate-100 border border-violet-500/30 hover:scale-105 hover:bg-violet-500/30 hover:border-violet-400/50 transition-all"
                          >
                            {s}
                          </span>
                        ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </section>

          {/* Projects */}
          <section id="projects" className="mt-8">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold text-white">Projects</h2>
              <div className="flex items-center gap-3">
                <input
                  value={projSearch}
                  onChange={(e) => setProjSearch(e.target.value)}
                  placeholder="Search projects..."
                  className="px-3 py-2 rounded-md bg-slate-700/30 border border-slate-600/40 text-slate-200 placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-violet-500/50"
                />
                <a
                  href={GITHUB_DISCOVER_URL}
                  target="_blank"
                  rel="noreferrer"
                  className="px-3 py-2 rounded-md bg-gradient-to-r from-violet-600 to-violet-500 hover:from-violet-500 hover:to-violet-400 text-white text-sm shadow-lg hover:shadow-violet-500/50 transition-all"
                >
                  Discover More
                </a>
              </div>
            </div>

            <div className="mt-4 flex flex-wrap gap-2">
              {projectsCategories.map((t) => (
                <button
                  key={t}
                  onClick={() => setProjTag(t)}
                  className={`px-3 py-1 rounded-full text-sm transition-all ${
                    projTag === t
                      ? "bg-violet-600 text-white shadow-lg shadow-violet-500/30"
                      : "bg-slate-700/30 text-slate-200 border border-slate-600/40 hover:bg-slate-700/50"
                  }`}
                >
                  {t}
                </button>
              ))}
            </div>

            <motion.div
              layout
              className="mt-6 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5"
            >
              <AnimatePresence>
                {filteredProjects.map((p) => (
                  <motion.article
                    key={p.id}
                    layout
                    initial={{ opacity: 0, y: 12 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: 12 }}
                    whileHover={{ translateY: -8 }}
                    transition={{ type: "spring", stiffness: 250, damping: 22 }}
                    className="relative rounded-2xl p-5 bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 shadow-lg hover:shadow-xl hover:shadow-violet-500/10 transition-shadow"
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div className="flex-1">
                        <h4 className="text-white font-semibold">{p.title}</h4>
                        <div className="text-xs text-slate-400 mt-1">
                          {p.tech} · {p.year}
                        </div>
                      </div>
                      <div className="text-xs text-violet-400 font-medium">
                        {p.tags.join(", ")}
                      </div>
                    </div>

                    <ul className="mt-3 text-slate-300 text-sm space-y-1">
                      {p.bullets.slice(0, 3).map((b, i) => (
                        <li key={i}>• {b}</li>
                      ))}
                    </ul>

                    <div className="mt-4 flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        {p.links.github && (
                          <a
                            href={p.links.github}
                            target="_blank"
                            rel="noreferrer"
                            onClick={(e) => e.stopPropagation()}
                            className="text-sm underline text-violet-400 hover:text-violet-300 transition"
                          >
                            Code
                          </a>
                        )}
                      </div>
                    </div>
                  </motion.article>
                ))}
              </AnimatePresence>
            </motion.div>

            {filteredProjects.length === 0 && (
              <p className="mt-6 text-center text-slate-400">
                No projects match your filters.
              </p>
            )}
          </section>

          {/* Certifications */}
          <section id="certifications" className="mt-8">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold text-white">
                Certifications & Courses
              </h2>
              <div className="flex items-center gap-3">
                <input
                  value={certSearch}
                  onChange={(e) => setCertSearch(e.target.value)}
                  placeholder="Search certifications..."
                  className="px-3 py-2 rounded-md bg-slate-700/30 border border-slate-600/40 text-slate-200 placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-violet-500/50"
                />
                <a
                  href={LINKEDIN_DISCOVER_URL}
                  target="_blank"
                  rel="noreferrer"
                  className="px-3 py-2 rounded-md bg-gradient-to-r from-violet-600 to-violet-500 hover:from-violet-500 hover:to-violet-400 text-white text-sm shadow-lg hover:shadow-violet-500/50 transition-all"
                >
                  Discover More
                </a>
              </div>
            </div>

            {/* Category Filters */}
            <div className="mt-4 flex flex-wrap gap-2">
              {certCategories.map((cat) => (
                <button
                  key={cat}
                  onClick={() => setCertTag(cat)}
                  className={`px-3 py-1 rounded-full text-sm transition-all ${
                    certTag === cat
                      ? "bg-violet-600 text-white shadow-lg shadow-violet-500/30"
                      : "bg-slate-700/30 text-slate-200 border border-slate-600/40 hover:bg-slate-700/50"
                  }`}
                >
                  {cat}
                </button>
              ))}
            </div>

            {/* Certifications Grid */}
            <motion.div
              layout
              className="mt-6 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5"
            >
              <AnimatePresence>
                {filteredCerts.map((cert) => (
                  <motion.article
                    key={cert.id}
                    layout
                    initial={{ opacity: 0, y: 12 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: 12 }}
                    whileHover={{ translateY: -8 }}
                    transition={{ type: "spring", stiffness: 250, damping: 22 }}
                    className="relative rounded-2xl p-5 bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 shadow-lg hover:shadow-xl hover:shadow-violet-500/10 transition-shadow"
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div className="flex-1">
                        <h4 className="text-white font-semibold">
                          {cert.title}
                        </h4>
                        <div className="text-xs text-violet-400 mt-1">
                          {cert.category}
                        </div>
                      </div>
                    </div>

                    <ul className="mt-3 text-slate-300 text-sm space-y-1">
                      {cert.bullets.slice(0, 3).map((b, i) => (
                        <li key={i}>• {b}</li>
                      ))}
                    </ul>
                  </motion.article>
                ))}
              </AnimatePresence>
            </motion.div>

            {filteredCerts.length === 0 && (
              <p className="mt-6 text-center text-slate-400">
                No certifications match your filters.
              </p>
            )}
          </section>

          {/* Contact */}
          <section
            id="contact"
            className="mt-10 bg-slate-800/40 backdrop-blur-sm rounded-2xl p-6 border border-slate-700/50 shadow-lg"
          >
            <h3 className="text-lg font-semibold text-white">Contact</h3>
            <p className="mt-3 text-slate-300">
              <a
                className="underline text-violet-400 hover:text-violet-300 transition"
                href="mailto:sondosahmed72@gmail.com"
              >
                sondosahmed72@gmail.com
              </a>{" "}
              | +20 127 661 1078
            </p>

            <div className="mt-4 flex flex-wrap gap-3">
              <a
                className="px-4 py-2 rounded-md bg-violet-600 hover:bg-violet-500 text-white shadow-lg hover:shadow-violet-500/50 transition-all"
                href="https://www.linkedin.com/in/sondos-ahmed-109787246/"
                target="_blank"
                rel="noreferrer"
              >
                LinkedIn
              </a>
              <a
                className="px-4 py-2 rounded-md bg-violet-600 hover:bg-violet-500 text-white shadow-lg hover:shadow-violet-500/50 transition-all"
                href="https://github.com/sondosahmed72"
                target="_blank"
                rel="noreferrer"
              >
                GitHub
              </a>
            </div>
          </section>

          <footer className="mt-10 text-center text-slate-400 text-xs">
            Built with React + Tailwind + Framer Motion — Elegant Dark Navy +
            Violet By Sondos Ahmed
          </footer>
        </div>
      </div>
    </div>
  );
}
