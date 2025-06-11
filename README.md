# cs189-homework-4-logistic-regression-with-newton-s-method-and-wine-classification-with-logistic-regr
**TO GET THIS SOLUTION VISIT:** [CS189-Homework 4 Logistic Regression with Newton’s Method and Wine Classification with Logistic Regr](https://mantutor.com/product/cs189-homework-4-logistic-regression-with-newtons-method-and-wine-classification-with-logistic-regression-solved/)


---

**For Custom/Order Solutions:** **Email:** mantutorcodes@gmail.com  

*We deliver quick, professional, and affordable assignment help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;60572&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;2&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (2 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;CS189-Homework 4 Logistic Regression with Newton’s Method and Wine Classification with Logistic Regression Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (2 votes)    </div>
    </div>
<h1>1&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Logistic Regression with Newton’s Method</h1>
Given examples x<sub>1</sub>,x<sub>2</sub>,…,x<em><sub>n </sub></em>∈ R<em><sup>d </sup></em>and associated labels <em>y</em><sub>1</sub>,<em>y</em><sub>2</sub>,…,<em>y<sub>n </sub></em>∈ {0,1}, the cost function for <em>unregularized </em>logistic regression is

<em><sup>n </sup></em>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 

X

<em>J</em>(w) , −&nbsp; <em>y</em><em>i </em>ln <em>s</em><em>i </em>+ (1 − <em>y</em><em>i</em>)ln(1 − <em>s</em><em>i</em>)

<em>i</em>=1

where <em>s<sub>i </sub></em>, <em>s</em>(x<em><sub>i </sub></em>· w), w ∈ R<em><sup>d </sup></em>is a weight vector, and <em>s</em>(γ) , 1/(1 + <em>e</em><sup>−γ</sup>) is the logistic function.

Define the <em>n </em>× <em>d </em>design matrix <em>X </em>(whose <em>i</em><sup>th </sup>row is x<sup>&gt;</sup><em><sub>i </sub></em>), the label <em>n</em>-vector y , [<em>y</em><sub>1 </sub>… <em>y<sub>n</sub></em>]<sup>&gt;</sup>, and s , [<em>s</em><sub>1 </sub>… <em>s<sub>n</sub></em>]<sup>&gt;</sup>. For an <em>n</em>-vector a, let lna , [ln<em>a</em><sub>1 </sub>… ln<em>a<sub>n</sub></em>]<sup>&gt;</sup>. The cost function can be rewritten in vector form as <em>J</em>(w) = −y · lns − (1 − y) · ln(1 − s).

<em>Hint: Recall matrix calculus identities</em><sub>x</sub>z y; ∇<sub>x</sub>f(y) =

; and(where<em>C </em>is a constant matrix).

<ul>
<li>Derive the gradient ∇<sub>w</sub><em>J</em>(w) of cost <em>J</em>(w) as a matrix-vector expression. Also derive <em>all intermediate derivatives </em>in matrix-vector form. Do NOT specify them in terms of their individual components.</li>
<li>Derive the Hessian) for the cost function <em>J</em>(w) as a matrix-vector expression.</li>
<li>Write the matrix-vector update law for one iteration of Newton’s method, substituting the gradient and Hessian of <em>J</em>(w).</li>
<li>You are given four examples x<sub>1 </sub>= [0.2 3.1]<sup>&gt;</sup>,x<sub>2 </sub>= [1.0 3.0]<sup>&gt;</sup>,x<sub>3 </sub>= [−0.2 1.2]<sup>&gt;</sup>,x<sub>4 </sub>= [1.0 1.1]<sup>&gt; </sup>with labels <em>y</em><sub>1 </sub>= 1,<em>y</em><sub>2 </sub>= 1,<em>y</em><sub>3 </sub>= 0,<em>y</em><sub>4 </sub>= 0. These points cannot be separated by a line passing through origin. Hence, as described in lecture, append a 1 to each x<em><sub>i</sub></em><sub>∈[4] </sub>and use a weight vector w ∈ R<sup>3 </sup>whose last component is the bias term (called α in lecture). Begin</li>
</ul>
&gt;

with initial weight <em>w</em><sup>(0) </sup>= <sup>h</sup>−1 1 0<sup>i </sup>. For the following, state only the final answer with four digits after the decimal point. You may use a calculator or write a program to solve for these, but do NOT submit any code for this part.

<ul>
<li>State the value of s<sup>(0) </sup>(the initial value of s).</li>
<li>State the value of w<sup>(1) </sup>(the value of w after 1 iteration).</li>
<li>State the value of s<sup>(1) </sup>(the value of s after 1 iteration).</li>
<li>State the value of w<sup>(2) </sup>(the value of w after 2 iterations).</li>
</ul>
<h1>2&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Wine Classification with Logistic Regression</h1>
The wine dataset data.mat consists of 6,497 sample points, each having 12 features. The description of these features is provided in data.mat. The dataset includes a training set of 6,000 sample points and a test set of 497 sample points. Your classifier needs to predict whether a wine is white (class label 0) or red (class label 1).

Begin by normalizing the data with each feature’s mean and standard deviation. You should use training data statistics to normalize both training and validation/test data. Then add a fictitious dimension. Whenever required, it is recommended that you tune hyperparameter values with crossvalidation.

Please set a random seed whenever needed and report it.

Use of automatic logistic regression libraries/packages is prohibited for this question. If you are coding in python, it is better to use scipy.special.expit for evaluating logistic functions as its code is numerically stable, and doesn’t produce NaN or MathOverflow exceptions.

<ul>
<li><em>Batch Gradient Descent Update</em>. State the batch gradient descent update law for logistic regression with `<sub>2 </sub> As this is a “batch” algorithm, each iteration should use <em>every training example</em>. You don’t have to show your derivation. You may reuse results from your solution to question 4.1.</li>
<li><em>Batch Gradient Descent Code</em>. Choose reasonable values for the regularization parameter and step size (learning rate), specify your chosen values, and train your model from question 3.1. Plot the value of the cost function versus the number of iterations spent in training.</li>
<li><em>Stochastic Gradient Descent (SGD) Update</em>. State the SGD update law for logistic regression with `<sub>2 </sub> Since this is not a “batch” algorithm anymore, each iteration uses <em>just one </em>training example. You don’t have to show your derivation.</li>
<li><em>Stochastic Gradient Descent Code</em>. Choose a suitable value for the step size (learning rate), specify your chosen value, and run your SGD algorithm from question 3.3. Plot the value of the cost function versus the number of iterations spent in training.</li>
</ul>
Compare your plot here with that of question 3.2. Which method converges more quickly? Briefly describe what you observe.

<ul>
<li>Instead of using a constant step size (learning rate) in SGD, you could use a step size that slowly shrinks from iteration to iteration. Run your SGD algorithm from question 3.3 with a step size <em><sub>t </sub></em>= δ/<em>t </em>where <em>t </em>is the iteration number and δ is a hyperparameter you select empirically. Mention the value of δ chosen. Plot the value of cost function versus the number of iterations spent in training.</li>
</ul>
How does this compare to the convergence of your previous SGD code?

<ul>
<li><em>Kaggle</em>. Train your <em>best </em>classifier on the entire training set and submit your prediction on the test sample points to Kaggle. As always for Kaggle competitions, you are welcome to add or remove features, tweak the algorithm, and do pretty much anything you want to improve your Kaggle leaderboard performance except that you may not replace or augment logistic regression with a wholly different learning algorithm. Your code should output the predicted labels in a CSV file.</li>
</ul>
Report your Kaggle username and your best score, and briefly describe what your best classifier does to achieve that score.

<h1>3&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; A Bayesian Interpretation of Lasso</h1>
Suppose you are aware that the labels <em>y<sub>i</sub></em><sub>∈[<em>n</em>] </sub>corresponding to sample points x<em><sub>i</sub></em><sub>∈[<em>n</em>] </sub>∈ R<em><sup>d </sup></em>follow the density law ,w) , <span style="text-decoration: line-through;">√</span>1&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <em>e</em>−(<em>y</em>−w·x)<sup>2</sup>/(2σ<sup>2</sup>) <em>f</em>(<em>y</em>|x

σ&nbsp;&nbsp;&nbsp; 2π

where σ &gt; 0 is a known constant and w ∈ R<em><sup>d </sup></em>is a random parameter. Suppose further that experts have told you that

<ul>
<li>each component of w is independent of the others, and</li>
<li>each component of w has the Laplace distribution with location 0 and scale being a known constant <em>b</em>. That is, each component w<em><sub>i </sub></em>obeys the density law <em>f</em>(w<em><sub>i</sub></em>) = <em>e</em><sup>−|w</sup><em><sup>i</sup></em><sup>|/<em>b</em></sup>/(2<em>b</em>).</li>
</ul>
Assume the outputs <em>y<sub>i</sub></em><sub>∈[<em>n</em>] </sub>are independent from each other.

Your goal is to find the choice of parameter w that is <em>most likely </em>given the input-output examples (x<em><sub>i</sub></em>,<em>y<sub>i</sub></em>)<em><sub>i</sub></em><sub>∈[<em>n</em>]</sub>. This method of estimating parameters is called <em>maximum a posteriori </em>(MAP); Latin for <em>“maximum [odds] from what follows.”</em>

<ol>
<li>Derive the <em>posterior </em>probability density law <em>f</em>(w|(x<sub>i</sub>,<em>y<sub>i</sub></em>)<em><sub>i</sub></em><sub>∈[<em>n</em>]</sub>) for w <em>up to a proportionality constant </em>by applying Bayes’ Theorem and using the densities <em>f</em>(<em>y<sub>i</sub></em>|x<sub>i</sub>,w) and <em>f</em>(w). Don’t try to derive an exact expression for <em>f</em>(w|(x<sub>i</sub>,<em>y<sub>i</sub></em>)<em><sub>i</sub></em><sub>∈[<em>n</em>]</sub>), as it is very involved.</li>
<li>Define the log-likelihood for MAP as `(w) , ln <em>f</em>(w|x<em><sub>i</sub></em><sub>∈[<em>n</em>]</sub>,<em>y<sub>i</sub></em><sub>∈[<em>n</em>]</sub>). Show that maximizing the MAP log-likelihood over all choices of w is the same as minimizing <sup>P<em>n</em></sup><em><sub>i</sub></em><sub>=1</sub>(<em>y<sub>i </sub></em>−w·x<sub>i</sub>)<sup>2 </sup>+λkwk<sub>1 </sub>where kwk<sub>1 </sub>= <sup>P<em>d</em></sup><em><sub>j</sub></em><sub>=1 </sub>|<em>w<sub>j</sub></em>| and λ is a constant.</li>
</ol>
<h1>4&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `<sub>1</sub>-regularization, `<sub>2</sub>-regularization, and Sparsity</h1>
You are given a design matrix <em>X </em>(whose <em>i</em><sup>th </sup>row is sample point x<em><sup>T</sup><sub>i </sub></em>) and an <em>n</em>-vector of labels y , [<em>y</em><sub>1 </sub>… <em>y<sub>n</sub></em>]<em><sup>T</sup></em>. For simplicity, assume <em>X </em>is whitened, so <em>X</em><sup>&gt;</sup><em>X </em>= <em>nI</em>. Do not add a fictitious dimension/bias term; for input 0, the output is always 0. Let x<sub>∗<em>i </em></sub>denote the <em>i</em><sup>th </sup>column of <em>X</em>.

<ol>
<li>The `<em><sub>p</sub></em>-norm for <em>w </em>∈ R<em><sup>d </sup></em>is defined as ||<em>w</em>||<em><sub>p </sub></em>= (<sup>P</sup><em><sub>i</sub><sup>d</sup></em><sub>=1 </sub>|<em>w<sub>i</sub></em>|<em><sup>p</sup></em>)<u><sup>1</sup></u><em><sup>p </sup></em>, where <em>p </em>&gt; 0. Plot the isocontours with <em>w </em>∈ R<sup>2</sup>, for the following:</li>
</ol>
(a) `<sub>0.5&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </sub>(b) `<sub>1&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </sub>(c) `<sub>2</sub>

Use of automatic libraries/packages for computing norms is prohibited for the question.

<ol start="2">
<li>Show that the cost function for `<sub>1</sub>-regularized least squares, <em>J</em><sub>1</sub>(w) , k<em>X</em>w − yk<sup>2 </sup>+ λkwk<sub>1 </sub>(where λ &gt; 0), can be rewritten as <em>J</em><sub>1</sub>(w) = kyk<sup>2 </sup>) where <em>f</em>(·, ·) is a suitable function whose first argument is a vector and second argument is a scalar.</li>
<li>Using your solution to part 2, derive necessary and sufficient conditions for the <em>i</em><sup>th </sup>component of the optimizer w<sup>∗ </sup>of <em>J</em><sub>1</sub>(·) to satisfy each of these three properties: <em>w</em><sup>∗</sup><em><sub>i </sub></em>&gt; 0,<em>w</em><sup>∗</sup><em><sub>i </sub></em>= 0, and <em>w</em><sup>∗</sup><em><sub>i </sub></em>&lt; 0.</li>
<li>For the optimizer w<sup># </sup>of the `<sub>2</sub>-regularized least squares cost function <em>J</em><sub>2</sub>(w) , k<em>X</em>w − yk<sup>2 </sup>+ λkwk<sup>2 </sup>(where λ &gt; 0), derive a necessary and sufficient condition for w<sup>#</sup><em><sub>i </sub></em>= 0, where w<sup>#</sup><em><sub>i </sub></em>is the <em>i</em>th component of w<sup>#</sup>.</li>
<li>A vector is called <em>sparse </em>if most of its components are 0. From your solution to part 3 and 4, which of w<sup>∗ </sup>and w<sup># </sup>is more likely to be sparse? Why?</li>
</ol>
