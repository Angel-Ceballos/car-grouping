# Car Grouping
## In this repo I will walk you through the process of creating a classificator for grouping cars based on their features.
-------------------------------------------
The idea is to explore the data, analyze and extract significat features for grouping based on a classification task.

Table of contents
-----------------
* [Data Exploration](#exploration)    
* [Prerequisite](#prerequisite)
* [Demo: YOLOv4](#yolov4)
* [Demo: Running the App](#app)

<a name="exploration"></a>
Data Exploration
------------
We will start by downloading the data through SSH and consequently exploring it.

1. Using SFTP, connect.
    
    ```shell
    $ sftp stdatalabelling.techchallenge.techchallenge@stdatalabelling.blob.core.windows.net
    ```

2. Type password: *NICKUeWQuX+O815kMn8BIgcx5rDHJCNA*. 

3. Download to local.
    
    ```shell
    $ get image_test.tar.gz
    ```
4. Explore data
    When reading all the images inside our directory, you can see that there are more than 11.5 k images without labels. So this is a problem, because labeling everything will take quite some time, so we need an strategy.
    * First we need to set an ideal number of images to create a subset. A reasonable number is **1k** images.
    * Now because we dont have pre-defined labels, we do not have a way of creating a balanced dataset taking into account the feature we want to detect/classify. So, given this issue, we will have to **randomly sample a subset** to obtain somewhat of a representative distrubution.

        ![Car subset](./display/output.png)

        **NOTE: I found the dataset using reverse image search but I assume that is not the goal so I will continue with the described strategy**
    * For creating the labels, I will be using an open source lib call **CVAT**.

5. Labeling strategy
    Before starting with the data labeling, lets use CVAT to explore our subset and create a strategy based on it. The idea is to create a superclass named vehicle with certain attributes. To be specific: color, type, and orientation. 

    - Color: I will be using the basic colors. Also to note, here we assume the car has a homogeneous paint. Some car may have a mixture of colors and this is something to consider during the creation of the output and the loss criteria. 
        - Red, green, yellow, orange, blue, violet, purple, black, white, brown, and gray.
    - Type: There are many types, some of them overlap, so lets use a simplified version.
        - Sport, hatchback, sedan, minivan, van, suv, pickup, truck, and bus.
    - Orientation: I believe a good approach would be to use cardinal points with respect to the camera and then transform to world frame. Although this solution would require more labeling time, so we will have to go with a simpler version.
        - Front, back, left, right





