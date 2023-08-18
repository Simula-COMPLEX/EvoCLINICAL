# EvoCLINICAL: Evolving Cyber-cyber Digital Twin with Active Transfer Learning for Automated Cancer Registry System


## Abstract

The Cancer Registry of Norway (CRN) collects information on cancer patients by receiving cancer messages from different medical entities (e.g., medical labs, hospitals) in Norway. Such messages are validated by an automated cancer registry system: GURI. Its correct operation is crucial since it lays the foundation for cancer research and provides critical cancer-related statistics to its stakeholders. Constructing a cyber-cyber digital twin (CCDT) for GURI can facilitate various experiments and advanced analyses of the operational state of GURI without requiring intensive interactions with the real system. However, GURI constantly evolves due to novel medical diagnostics and treatment, technological advances, etc. Accordingly, CCDT should evolve as well to synchronize with GURI. A key challenge of achieving such synchronization is that evolving CCDT needs abundant data labelled by the new GURI. To tackle this challenge, we propose EvoCLINICAL, which considers the CCDT developed for the previous version of GURI as the pretrained model and fine-tunes it with the dataset labelled by querying a new GURI version. EvoCLINICAL employs a genetic algorithm to select an optimal subset of cancer messages from a candidate dataset and query GURI with it. We evaluate EvoCLINICAL on three evolution processes. The precision, recall, and F1 score are all greater than 91%, demonstrating the effectiveness of EvoCLINICAL. Furthermore, we replace the active learning part of EvoCLINICAL with random selection to study the contribution of transfer learning to the overall performance of EvoCLINICAL. Results show that employing active learning in EvoCLINICAL increases its performances consistently.



![image](https://github.com/Simula-COMPLEX/EvoCLINICAL/blob/main/figures/overview.png)
