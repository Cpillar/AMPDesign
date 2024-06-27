## **About AMPDesign**

Antimicrobial resistance poses a threat to human well-being, especially with the emergence of multidrug-resistant bacteria. Antimicrobial peptides (AMPs), as one of the most promising alternatives to traditional antibiotics, have attracted significant attention in recent years. However, designing AMPs using traditional experimental methods is time-consuming and labor-intensive. Therefore, in this study, **we propose an out-of-the-box AMP design toolbox, named AMPDesign**. AMPDesign integrates strategies from **Generative Adversarial Networks (GANs)** and **Reinforcement Learning**, and incorporates our previously developed activity predictor to generate AMPs with high activity levels. Notably, AMPDesign is the first AMP generation toolbox capable of designing AMPs against specific bacterial species.



![img](https://awi.cuhk.edu.cn/~dbAMP/assets/img/generation_fig3.png)

![img](https://awi.cuhk.edu.cn/~dbAMP/assets/img/generation_fig4.png)



**Results**

We designed AMPs targeting E. coli using AMPDesign. We used experimentally validated anti-E. coli sequences from the dbAMP as the training set and generated 100 new peptide sequences. To validate the antibacterial activity of the generated sequences, we used the published AMPActiPred tool to predict their antibacterial activity. The prediction results showed that 82 out of the 100 generated sequences had antibacterial activity against E. coli.

We employed several commonly used peptide descriptors to observe the distribution of the newly generated sequences in the latent space compared to the original sequences in the training set. These peptide descriptors reflect the compositional information and physicochemical properties of the peptide sequences. After dimensionality reduction using UMAP, we can easily see that the newly designed AMPs had a similar distribution to the original AMPs **[Figure 1 (A)]**. This indicates that AMPDesign effectively learned the sequence characteristics of the original dataset and used the learned sequence patterns to generate new AMPs.

![img](https://awi.cuhk.edu.cn/~dbAMP/assets/img/generation_fig1.png)

**Figure 1**

**Figure 1 (B-I)** presents the distribution of common physicochemical properties of the newly generated sequences compared to the original dataset sequences. It is clear that the physicochemical property distribution of the **82 newly designed AMPs** closely resembles that of the original dataset. Finally, we named the new sequences **dbAMP_G0001** to **dbAMP_G0082** for further research by users.

![img](https://awi.cuhk.edu.cn/~dbAMP/assets/img/generation_fig2.png)

**Figure 2**

Previous literature has pointed out that lipopolysaccharides are one of the targets of Gram-negative bacteria. To further investigate the effectiveness of AMPDesign, we conducted a case study using docking. We selected dbAMP_06098, an anti-E. coli sequence from dbAMP, as the reference sequence and docked it with lipopolysaccharides. Additionally, we randomly selected five newly generated AMPs and docked them with lipopolysaccharides. The results, shown in **Figure 2**, indicate that docking results of new sequences were very close to those of dbAMP_06098. This further demonstrates that AMPDesign can effectively design AMPs, and we believe AMPDesign can become a valuable tool in the field of drug design.# AMPDesign

