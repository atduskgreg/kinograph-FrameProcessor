/*

This sketch extracts film frames from scans.
It expects there to be multiple frames visible in the scan
with only the center one targeted for extraction.
It uses the location of the sprocket holes to determine the top and bottom of the frame.
It expects all sprocket holes fully visible within the central target region to
be associated with the target frame.
It uses vertical contours in the image to find the sides of the frame.

Adjustable variables:

resizedImageWidth - resize original image to this width (keep proportions). bigger is slower. IMPORTANT: determines output image resolution.
roiTop - the top of the area to look for the frame (must include the frame's top sprocket and no above full sprocket)
roiHeight - the height of the area to look for the frame (must be tall enough to include the frame's bottom sprocket and no full lower sprockets)
searchColumn - which column in the image to look for the sprockets
distanceBetweenSprockets - number of pixels between sprockets
minVerticalEdgeLength - adjust down if it crashes looking for the side of the frame

Detection technique:

* The image is loaded up and resized.
* A Region of Interest is defined in the center of the image
* To find the top and bottom of the frame:
** Find y-dimension sobel edges in the ROI
** Threshold sobel image
** Look at each row of the image at the searchColumn y-position
** If the pixel is white, we've found the edge of a sprocket
** Use the distance between the sprocket lines to ensure we've only found complete sprockets (ones with both top and bottom)
** Top of top sprocket is the top of the frame, bottom of bottom sprocket is the bottom of the frame
* To find the left and right sides of the frame:
** Find x-dimension sobel edges in the ROI
** Threshold sobel image
** Find contours in the sobel image
** Remove short contours (those shorter than minVerticalEdgeLength)
** Calculate polygon approximations of contours to turn them into straight lines
* Calculate coordinates of frame based on frame edges, top and bottom
* Copy out portion of source image based on these coordinates

*/

import gab.opencvpro.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.CvType;

import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;

import org.opencv.core.Point;

import java.awt.Rectangle;

int resizedImageWidth = 500;
int roiTop = 270;
int roiHeight = 300;
int searchColumn = 70;
int distanceBetweenSprockets = 32;
int minVerticalEdgeLength = 175;

OpenCVPro sprocketProcessor, edgeProcessor;
PImage src, dst, dst2, output;

Rectangle selectedArea;

ArrayList<MatOfPoint> contours;
ArrayList<MatOfPoint2f> approximations;

class Sprocket {
  int top;
  int bottom;

  Sprocket(int top, int bottom) {
    this.top = top;
    this.bottom = bottom;
  }
}

ArrayList<Sprocket> sprockets;

Rectangle roi;


void setup() {
  // load source image and resize it
  src = loadImage("_MG_9859.JPG");
  src.resize(resizedImageWidth, 0);

  size(src.width*2, src.height);

  // setup ROI
  roi = new Rectangle(0, roiTop, src.width, roiHeight);

  // create opencv object for finding top and
  // bottom of frame and load original image into it
  sprocketProcessor = new OpenCVPro(this, src.width, src.height);
  sprocketProcessor.loadImage(src);
  // convert the original image to gray for processing
  sprocketProcessor.gray();
  
  // extract a submat based on the ROI and set it to
  // be the one we really work with
  Mat grayMat = sprocketProcessor.getBufferGray();
  Mat roiMat = grayMat.submat(roi.y, roi.y+roi.height, roi.x, roi.x+roi.width );
  sprocketProcessor.setBufferGray(roiMat);
  
  // filter image to bring out horizontal lines
  // and binarize it
  sprocketProcessor.findSobelEdges(0, 2);
  sprocketProcessor.threshold(50);
  
  // create opencv object for finding left and right of frame
  edgeProcessor = new OpenCVPro(this, roi.width, roi.height);
  // load a copy of the gray but otherwise unfiltered roi into it
  edgeProcessor.setBufferGray(roiMat.clone());
  // filter image to bring out vertical lines
  // and binarize it
  edgeProcessor.equalizeHistogram();
  edgeProcessor.findSobelEdges(2, 0);
  edgeProcessor.threshold(150);

  //Imgproc.dilate(opencv2.getBufferGray(), opencv2.getBufferGray(), new Mat());

  // Convert processed sprocket image into a PImage
  // so we can do pixel operations on it
  dst = createImage(src.width, src.height, ARGB);
  sprocketProcessor.toPImage(sprocketProcessor.getBufferGray(), dst);

  // === BEGIN FIND TOP AND BOTTOM SPROCKETS ===
  
  // find all sprocket edges
  // loop through rows of image one at a time
  // if this row has a white pixel at the searchColumn
  // and the previous row did not
  // then this row is a sprocket edge
  ArrayList<Integer> sprocketEdges = new ArrayList<Integer>();

  for (int row = 1; row < dst.height; row++) {
    int prevI = searchColumn + (row-1)*dst.width;
    int thisI = searchColumn + row*dst.width;

    boolean thisRow = brightness(dst.pixels[thisI]) > 0 ;
    boolean prevRow = brightness(dst.pixels[prevI]) > 0;

    if (thisRow && !prevRow) {
      sprocketEdges.add(row);
    }
  }

  // Now we need to combine the sprocketEdges into 
  // sprockets based on the distance between them.
  // loop over them and ad adjcacent edges as a
  // sprocket only if the distance between them is
  // less than our threshold.
  // Key idea: distance between sprocket holes
  // is larger than height of individual sprocket hole.
  sprockets = new ArrayList<Sprocket>();

  // loop over sprocketEdges and determine pairs
  for (int i = 1; i < sprocketEdges.size(); i++) {
    int prev = sprocketEdges.get(i-1);
    int curr = sprocketEdges.get(i);

    if (abs(curr - prev) < distanceBetweenSprockets) {
      sprockets.add(new Sprocket(prev, curr));
    }
  }

  // first sprocket is the top one
  // last is the bottom one
  Sprocket topSprocket = sprockets.get(0);
  Sprocket bottomSprocket = sprockets.get(sprockets.size()-1);
  
  // === BEGIN FIND LEFT AND RIGHT FRAME EDGES ===

  // find contours in the filtered edge image
  contours = new ArrayList<MatOfPoint>();
  Imgproc.findContours(edgeProcessor.getBufferGray(), contours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_NONE);

  // get rid of contours that are shorter than minVerticalEdgeLength
  contours = filterContours(contours);

  // convert the remaining contours into polygon approximations
  // i.e. straight lines
  approximations = createPolygonApproximations(contours);

  // the x-positions of these two are the left and right edges
  // of the frame
  float frameRight = (float)approximations.get(0).toArray()[0].x;
  float frameLeft = (float)approximations.get(1).toArray()[0].x;

  // === CALCULATE THE FRAME LOCATION and EXTRACT IT ===
  
  // make a rectangle starting at frameLeft and topSprocket.top
  // and extending the width and height of the frame
  selectedArea = new Rectangle((int)frameLeft, topSprocket.top, (int)(frameRight-frameLeft), bottomSprocket.bottom-topSprocket.top);
  // create a PImage for output and copy
  // the pixels from the source image into it.
  // NB: have to adjust the y-position down by the y-position of the ROI
  output = createImage(selectedArea.width, selectedArea.height, ARGB);
  output.copy(src, selectedArea.x, roi.y + selectedArea.y, selectedArea.width, selectedArea.height, 0,0,selectedArea.width, selectedArea.height);
  
  // convert edgeProcessor image into PImage for display (debugging only)
  dst2 = createImage(src.width, src.height, ARGB);
  edgeProcessor.toPImage(edgeProcessor.getBufferGray(), dst2);
}


// === HELPER FUNCTIONS ===

ArrayList<MatOfPoint> filterContours(ArrayList<MatOfPoint> cntrs) {
  ArrayList<MatOfPoint> result = new ArrayList<MatOfPoint>();
  for (MatOfPoint contour : cntrs) {
    if (contour.toArray().length > minVerticalEdgeLength) {
      result.add(contour);
    }
  }
  return result;
}

ArrayList<MatOfPoint2f> createPolygonApproximations(ArrayList<MatOfPoint> cntrs) {
  ArrayList<MatOfPoint2f> result = new ArrayList<MatOfPoint2f>();

  double epsilon = cntrs.get(0).size().height * 0.01;

  for (MatOfPoint contour : cntrs) {
    MatOfPoint2f approx = new MatOfPoint2f();
    Imgproc.approxPolyDP(new MatOfPoint2f(contour.toArray()), approx, epsilon, true);
    result.add(approx);
  }

  return result;
}

void drawContours(ArrayList<MatOfPoint> cntrs) {
  for (MatOfPoint contour : cntrs) {
    beginShape();
    Point[] points = contour.toArray();
    for (int i = 0; i < points.length; i++) {
      vertex((float)points[i].x, (float)points[i].y);
    }
    endShape();
  }
}

void drawContours2f(ArrayList<MatOfPoint2f> cntrs) {
  for (MatOfPoint2f contour : cntrs) {
    beginShape();
    Point[] points = contour.toArray();

    for (int i = 0; i < points.length; i++) {
      vertex((float)points[i].x, (float)points[i].y);
    }
    endShape(CLOSE);
  }
}
void draw() {
  image(src, 0, 0);
  
  pushMatrix();
  scale(0.75);
  image(dst, src.width * 4.0/3, 0);
  image(dst2, src.width * 4.0/3, roi.height + 10);
  popMatrix();
  
  image(output,src.width + 10, height-output.height-10);
  noFill();
  strokeWeight(4);

  stroke(255, 0, 0);
  rect(roi.x, roi.y, roi.width, roi.height);
  strokeWeight(1);

  stroke(0, 0, 255);
  translate(roi.x, roi.y);
  line(searchColumn, 0, searchColumn, src.height);

  for (Sprocket sprocket : sprockets) {
    stroke(0, 255, 0);
    line(0, sprocket.top, src.width, sprocket.top);
    stroke(255, 0, 0);
    line(0, sprocket.bottom, src.width, sprocket.bottom);
  }

  stroke(255);
  strokeWeight(3);
  drawContours2f(approximations);

  strokeWeight(5);
  stroke(0, 255, 0);
  rect(selectedArea.x, selectedArea.y, selectedArea.width, selectedArea.height);
}

