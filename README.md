<a name="readme-top"></a>

<div align="center">
    <h1 align="center">Math is Cool, I Promise</h1>
    <p align="center">
        This is a collection of projects that I've worked on that use mathematical tools
        to solve interesting problems in a variety of fields by Dallin Stewart
    </p>
</div>

<hr>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#welcome">Welcome</a></li>
    <li><a href="#sound">Analyzing Sound Waves</a></li>
    <li><a href="#stability">How Your Computer Lies to You</a></li>
    <li><a href="#recognition">Facial Recognition without Machine Learning</a></li>
    <li><a href="#digits">Classifying Handwritten Digits without ML</a></li>
    <li><a href="#compression">Image Compression</a></li>
    <li><a href="#segmentation">Image Segmentation</a></li>
    <li><a href="#optimization">How to Optimize Your Nutrition</a></li>
    <li><a href="#ranking">Google's Page Rank Algorithm</a></li>
    <li><a href="#recommender">Recommender System without Machine Learning</a></li>
    <li><a href="#markov">How to Talk like Yoda</a></li>
  </ol>
</details>

<!-- Welcome -->
## Welcome

This repository is a collection of projects that I started as an assignment for my math classes that is still 
under development. Credit for a lot of the ideas in these projects goes to the 
<a href='https://acme.byu.edu/2022-2023-materials'>ACME</a>
program at BYU. Each of the projects are distinct, but most use tools from linear algebra, calculus, and probability 
including:
    <li>Convex Optimization</li>
    <li>Singular Value Decomposition</li>
    <li>Eigenfaces Method</li>
    <li>Least Squares</li>
    <li>Adjacency Matrices and Graph Theory</li>
    <li>Breadth First Search</li>
    <li>Nearest Neighbors</li>
    <li>Markov Chains</li>
    <li>Monte Carlo Methods</li>
    <li>Interpolation</li>
    <li>Fast Fourier Transform</li>

### <a href="https://acme.byu.edu/">Applied and Computational Math Emphasis</a>

Brigham Young University’s Applied and Computational Math Emphasis (ACME) program is a major designed for solving the 
problems of the 21st century. Mathematics provides the foundation of modern technology and science, it is the key to 
building successful algorithms in artificial intelligence and machine learning, and it provides the analytical power 
needed to process, evaluate, and take full advantage of the ever-growing flood of data and information in the world. 
ACME is a new educational model that teaches both the theory and the practical skills in mathematics, statistics, and 
computation needed to solve the problems of the modern world.

<hr>

## Project Descriptions

### <a name="sound">Analyzing Sound Waves</a>
This project uses convolution ad the Fast Fourier Transform to analyze sound waves and images. I used it to modify the 
sound of a piano playing to make it sound like it is being played in a large concert hall. I also used it remove noise 
from an image of a license plate so that critical information would be easier to read..

### <a name="stability">How Your Computer Lies to You</a>
This project uses theory from conditioning and stability of computer algorithms to show how your computer can lie to you.
Because computers represent numbers in a finite number of bits, they have to approximate numbers with too many significant 
digits, and can only represent numbers in a finite range. This means that when you perform a calculation on a computer, 
the result may not be exactly what you expect.

### <a name="recognition">Facial Recognition without Machine Learning</a>
This project uses the Eigenfaces method to recognize faces without using machine learning. The Eigenfaces method involves 
using the singular value decomposition to find the principal components of a set of images. The principal components are 
the directions in which the images vary the most. It first computes the mean image, and then subtracts the mean from each 
image to find the most salient features of each image. This allows the program to find the most similar images overall, 
while only using linear algebra!

### <a name="digits">Classifying Handwritten Digits without ML</a>
This project uses the nearest neighbors method to classify handwritten digits without using machine learning. The nearest 
neighbors method involves finding the k nearest neighbors to a point, and then classifying the point based on the class 
of the nearest neighbors. In this project, I used it to classify handwritten digits. I used the MNIST dataset, which is a 
dataset of 70,000 handwritten digits. I used the first 60,000 digits to train the model, and the last 10,000 digits to test 
the model. I found that I could classify the digits with about 90% accuracy.

### <a name="compression">Image Compression</a>
This project uses the singular value decomposition to compress images. The singular value decomposition is a powerful 
tool that can be used to find the principal components of a matrix. In this project, I used it to find the most important 
elements of the eigenbasis of an image. I then used the eigenbasis to reconstruct the image, and found that I could 
compress the image by 90% without losing much of the original information.

### <a name="segmentation">Image Segmentation</a>
This project uses the singular value decomposition to segment images.

### <a name="optimization">How to Optimize Your Nutrition</a>
This projects uses the simplex method to optimize your nutrition. The simplex method is a powerful tool for solving 
linear programming problems. It is used to find the optimal solution to a linear programming problem. In this project, 
I used it to find the optimal combination of foods to eat to maximize the amount of nutrients you get while staying 
within your daily calorie limit.

### <a name="ranking">Google's Page Rank Algorithm</a>
This project uses the Page Rank algorithm to rank web pages. The Page Rank algorithm is a powerful tool for ranking 
a list of options using graph theory and adjacency matrices. In this project, I used it to ranks a small list of stanford's
websites and to determine the best team to play in the first round of the NCAA tournament.

### <a name="recommender">Recommender System without Machine Learning</a>
This project uses non-negative matrix factorization to create a recommender system without using machine learning. The 
non-negative matrix factorization is a powerful tool.

### <a name="markov">How to Talk like Yoda</a>
This project uses Markov chains to create a Markov chain that can generate text that sounds like Yoda. Markov chains are 
a powerful tool for modeling random processes. In this project, I used it to model the way Yoda speaks. I used a dataset 
of all of Yoda's lines from the Star Wars movies, and then used the Markov chain to generate new lines that sound like 
Yoda.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<hr>

<!-- CONTACT -->
## Contact

Dallin Stewart - dallinpstewart@gmail.com

[![LinkedIn][linkedin-icon]][linkedin-url]
[![GitHub][github-icon]][github-url]
[![Email][email-icon]][email-url]


<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- MARKDOWN LINKS & IMAGES -->
[Python-icon]: https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white
[Python-url]: https://www.python.org/

[NumPy-icon]: https://img.shields.io/badge/NumPy-2596be?style=for-the-badge&logo=numpy&logoColor=white
[NumPy-url]: https://numpy.org/

[Pandas-icon]: https://img.shields.io/badge/Pandas-120756?style=for-the-badge&logo=pandas&logoColor=white
[Pandas-url]: https://pandas.pydata.org/


[linkedIn-icon]: https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white
[linkedIn-url]: https://www.linkedin.com/in/dallinstewart/

[github-icon]: https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white
[github-url]: https://github.com/binDebug3

[Email-icon]: https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white
[Email-url]: mailto:dallinpstewart@gmail.com
