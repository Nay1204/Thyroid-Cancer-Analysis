# Thyroid-Cancer-Analysis
In this project, we explore the use of deep learning and machine learning
techniques for the classification of thyroid disease from medical images. Our
approach leverages transfer learning by utilizing a pre-trained VGG16 model to
extract high-level features from thyroid ultrasound images. To address class
imbalance in the dataset, we applied SMOTE (Synthetic Minority Oversampling
Technique), along with image augmentation techniques such as rotation, flipping,
and contrast adjustments. The extracted features were classified using a dense
neural network, achieving promising accuracy. Label encoding and one-hot
encoding were used for efficient label handling, while performance was evaluated
using metrics like accuracy, precision, recall, and F1-score. This system
demonstrates the potential of automated, image-based classification as a supportive
tool for early detection of thyroid abnormalities, reducing reliance on manual
interpretation and enhancing diagnostic efficiency.
While the model achieved strong classification performance, challenges like class
imbalance and limited data were addressed using SMOTE and image augmentation
techniques. Hyperparameter tuning further improved the model’s reliability and
reduced false positives. Though focused on model development, this work sets the
foundation for future integration into a clinical interface. The project emphasizes
the effectiveness of AI-driven image analysis in thyroid disease detection, offering
a scalable tool for faster, more accurate diagnostics.
