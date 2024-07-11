```mermaid
graph TD
    A[Input Image] --> B[Data Preprocessing]
    B --> C{Model Selector}
    C -->|ResNet| D[ResNet Model]
    C -->|MobileNet| E[MobileNet Model]
    D --> F[Age Prediction]
    E --> F
    F --> G[Post-processing]
    G --> H[Final Age Prediction]

    subgraph "Common Interface"
    I[AgePredictor Abstract Class]
    end

    subgraph "Model Implementation"
    J[ResNetAgePredictor]
    K[MobileNetAgePredictor]
    end

    I -.-> J
    I -.-> K
    J -.-> D
    K -.-> E

    subgraph "Input Standardization"
    L[Image Resizing]
    M[Pixel Normalization]
    end

    B --> L
    B --> M

    subgraph "Output Standardization"
    N[Age Range Clipping]
    O[Uncertainty Estimation]
    end

    F --> N
    F --> O

    subgraph "Training Pipeline"
    P[Data Augmentation]
    Q[Transfer Learning]
    R[Fine-tuning]
    S[Performance Evaluation]
    end

    P --> Q
    Q --> R
    R --> S
  ```
