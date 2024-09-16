# Graph GAN: A Generative AI Model Utilizing Wasserstein GAN

Graph GAN is a generative model designed to create graph-structured data using Wasserstein GAN (WGAN) principles. The project focuses on addressing the challenges associated with generating complex graph structures, which are commonly found in fields like circuit design, social networks, and biological systems.

### Key Features:
- **Generative Model**: Uses a Wasserstein GAN framework to generate realistic graph structures.
- **Graph Representation**: The model efficiently handles graph-structured data, preserving node connections, and edge relationships.
- **Improved Training Stability**: By utilizing the Wasserstein distance, the model ensures more stable training compared to traditional GANs, reducing mode collapse and improving convergence.
- **Applications**: This approach is well-suited for generating synthetic benchmark circuits, modeling network traffic, and other applications where graph data is crucial.

### Project Objectives:
1. **Graph Data Generation**: Develop a robust GAN model to generate realistic and meaningful graph structures.
2. **Stability in GAN Training**: Leverage the WGAN framework to address instability issues in training.
3. **Domain-Specific Applications**: Tailor the model for use in fields like VLSI circuit design, with future adaptability to other graph-driven domains.

### Future Enhancements:
- Add support for conditional graph generation.
- Extend the model for more complex graph types such as directed or weighted graphs.
- Implement advanced evaluation metrics for graph similarity.

---

### The dataset used for traning the GraphGAN is available in the SURI Project and this this project is a part of Artificial netlist generation project by the VDA Lab at Arizona State University
