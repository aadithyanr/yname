# YC name-model training results

Two small character MLPs were trained from scratch. No LLM or pretrained model was used.

| Model | Parameters | Best step | Validation loss | Test loss | Test perplexity |
|---|---:|---:|---:|---:|---:|
| Plain | 46,700 | 1,000 | 2.3835 | 2.3790 | 10.79 |
| Conditional | 49,068 | 1,000 | 2.3956 | 2.3805 | 10.81 |

## Plain model samples

Commeting, Mender, Instersebio, Pentence, Coperal, Sentrive, Trebous, Mind ai, Nopernase, Cundre, Findecond, Coutze robotics, Menterch, Trandet, Pasentra, Seneriant, Dreambai, Signamedics, Intrance, Drebon, Converch, Coservel theraper, Striam, Kend bioscience

## Category-conditioned samples

### B2B

Codera, Flowaflow, Spackitt, Archardets, Sharblet, Comprade, Calvera, Flight, Instara, Airstite, Prondusting, Landwork

### Consumer

Spackit, Codero, Hopplater, Flowdy, Clavera, Packan, Shippi, Pockant, Clowdrout, Codeto, Hiffoct, Spockitus

### Education

Codestove, Coderocode, Compure, Packit, Sunflex, Educkor, Flockerut, Sairmand, Contus, Perenter, Cademock, Code airas

### Fintech

Packit, Chardstate, Carver, Packor, Packen, Manker, Instackd, Arkana, Packispark, Wovencare, Codemand, Panari

### Healthcare

Reve health, Convertet healthcare, Aghion biosciences, Salar health, Pronchel biotych, Brickit, Provel health, Zent health, Pinse bioncerous, Sente labs, Airnal health, Heall health

### Industrials

Coreboord, Moro robotics, Spackate, Starabourd, Arrana, Mentris, Onstroom, Bristint, Andarcone, Perap redotics, Buildsater, Carbles

### Other

Mandstack, Compurlity, Flowdatum, Traver industries, Velic labs, Modingumedics, Mackit, Instivent, Hemone rodical, Velagion, Claventre stace, Prandane

### Real Estate and Construction

Chardesurids, Parchive, Codentork, Sparbascience, Brianclow, Compun industries, Bulldo, Hemand, Riveld, Compoo robotics, Insperp, Porpactround

## Notes

- Source records: 6,194
- Unique cleaned names: 6,090
- Training/validation/test names: 4,874/608/608
- Generated samples shown above exclude exact matches to every known directory name.
- Candidates at or above 0.84 similarity to a known name are also rejected (0.80 for very short names).
- YC's sparse Government and Unspecified labels are merged into Other for conditioning.
- Lower loss is better; perplexity is the model's average effective number of next-character choices.
