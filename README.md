# Total Perspective Vortex

## Description

### Goal
- Process EEG datas (parsing and filtering)
- Implement a dimensionality reduction algorithm
- Use the pipeline object from scikit-learn
- Classify a data stream in "real time"

#### Architecture

```mermaid
graph TD
    A[PhysioNet Data download] --> B[EDF file read & concat]
    B --> C[Channel name standardization]
    C --> D[10-20 montage setting]
    D --> E[Annotation renaming T1→hands, T2→feet]
    E --> F[Average reference setting projection]
    F --> G[7-30 Hz band filtering]
    G --> H[Epoch extraction -1~4sec]
    H --> I[Learning segment cropping 1~2sec]
    I --> J["X: feature matrix, y: label array"]
```

### Cite
Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N., Wolpaw, J.R. BCI2000: A General-Purpose Brain-Computer Interface (BCI) System. IEEE Transactions on Biomedical Engineering 51(6):1034-1043, 2004.