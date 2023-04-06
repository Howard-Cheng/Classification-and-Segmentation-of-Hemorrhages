# Classification-and-Segmentation-of-Hemorrhages
## Abstract
In order to obtain a machine learning algorithm capable of classifying hemorrhages from different scanning windows, three
different methods were developed and compared. These included logistic regression, transfer learning CNN, tree-based models.
Additionally, two methods were used in order to segment the images; the first of which being an original method that allows
for the use of linear regression in segmentation, and the second being a conventional method. These are all elaborated upon in
the Models section of this paper below. Overall, the Neural Network produced the best results for classification.

## Introduction
Brain hemorrhages, or cerebral hemorrhages, are an emergency condition in which a blood vessel within the brain ruptures
causing internal bleeding [1]. They are most commonly the result of trauma and elevated blood pressure, and can lead to
strokes, loss of brain function, and death [2]. For this reason, it is incredibly important to be able to classify and locate
hemorrhages in the brain in order to provide proper treatment as fast as possible. One such avenue for this is through the use
of image classification and segmentation modeling, with hopes of producing high-accuracy models that can assist a doctor in
diagnosing a hemorrhage. As such, during this report we will go over different types of machine learning (ML) models that we
developed and their accuracy scores, including linear and logistic regression, convolutional neural networks (CNNs), decision
trees, and image segmentation
