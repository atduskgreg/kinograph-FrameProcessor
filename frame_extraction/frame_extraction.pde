/*

 OpenCVPro compatibility - 0.2.1

 This sketch extracts film frames from scans.
 It expects there to be multiple frames visible in the scan
 with only the center one targeted for extraction.
 It uses the location of the sprocket holes to determine the top and bottom of the frame.
 It expects all sprocket holes fully visible within the central target region to
 be associated with the target frame.
 It uses vertical contours in the image to find the sides of the frame.
 
 Adjustable variables:
 
 inputFilename - name of input scanned frame to be processed (assume to be in data/ folder)
 resizedImageWidth - resize original image to this width (keep proportions). bigger is slower. IMPORTANT: determines output image resolution.
 roiTop - the top of the area to look for the frame (must include the frame's top sprocket and no above full sprocket)
 roiHeight - the height of the area to look for the frame (must be tall enough to include the frame's bottom sprocket and no full lower sprockets)
 searchColumn - which column in the image to look for the sprockets
 distanceBetweenSprockets - number of pixels between sprockets
 minVerticalEdgeLength - adjust down if it crashes looking for the side of the frame
 frameWidth - the width of the output frame in pixels (as observed after resizing the image to resizedImageWidth)
 frameHeight - the height of the output frame in pixels (as observed after resizing the image to resizedImageWidth)
 
 Detection technique:
 
 * The image is loaded up and resized.
 * A Region of Interest is defined in the center of the image
 * To find the top and bottom of the frame:
 ** Find y-dimension sobel edges in the ROI
 ** Threshold sobel image
 ** Look at each row of the image at the searchColumn y-position
 ** If the pixel is white (and the one above it was not), we've found the edge of the top sprocket
 ** Top of top sprocket is the top of the frame, measure the fixed frameHeight from there to find the bottom of the frame
 * To find the left and right sides of the frame:
 ** Find x-dimension sobel edges in the ROI
 ** Threshold sobel image
 ** Find contours in the sobel image
 ** Remove short contours (those shorter than minVerticalEdgeLength)
 ** Calculate polygon approximations of contours to turn them into straight lines
 * Find the rightmost frame edge candidate (that's to the left of the sprocket searchColumn)
 * Measure the fixed frameWidth left to determine the left side of the frame
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

String inputFilename = "_MG_1267.JPG";
int resizedImageWidth = 1107; // to scale to 640x480 frame size
int roiTop = 485;
int roiHeight = 605;
int searchColumn = 976;
int distanceBetweenSprockets = 78;
int minVerticalEdgeLength = 150;
int frameWidth = 640;
int frameHeight = 480;

OpenCVPro opencv;
PImage src, dst, dst2, output;

Rectangle selectedArea;

ArrayList<Contour> contours;
ArrayList<Contour> approximations;

Rectangle roi;

void setup() {
  // load source image and resize it
  src = loadImage(inputFilename);
  src.resize(resizedImageWidth, 0);

  size(src.width, src.height/2);

  // setup ROI
  roi = new Rectangle(0, roiTop, src.width, roiHeight);

  // create opencv object for finding top and
  // bottom of frame and load original image into it
  opencv = new OpenCVPro(this, src.width, src.height);
  opencv.loadImage(src);
  // convert the original image to gray for processing


  opencv.setROI(roi.x, roi.y, roi.width, roi.height);
  PImage gray = opencv.getSnapshot(); // before doing the sprocket processing

  // filter image to bring out horizontal lines
  // and binarize it
  opencv.findSobelEdges(0, 1);  
  opencv.threshold(30);
  opencv.dilate();

  dst = opencv.getROI();

  // reset back to the original gray image
  opencv.loadImage(gray);
  opencv.setROI(roi.x, roi.y, roi.width, roi.height);

  // filter image to bring out vertical lines
  // and binarize it
  opencv.equalizeHistogram();
  opencv.findSobelEdges(2, 0);    
  opencv.threshold(100);
  dst2 = opencv.getROI();

  // === BEGIN FIND TOP SPROCKETS EDGE ===
  // it's the topmost white pixel in the searchColumn

  int topSprocketEdge = 0;
  
  for (int row = 1; row < dst.height; row++) {
    int i = searchColumn + row*dst.width;

    if (brightness(dst.pixels[i]) > 0) {
      topSprocketEdge = row;
      break;
    }
  }
    
  // === BEGIN FIND RIGHT FRAME EDGE ===

  // find contours in the filtered edge image
  contours = opencv.findContours();
  approximations = new ArrayList<Contour>();

  // make polygon approximations of contours
  // that are not too short (i.e. have too few points)
  for (Contour c : contours) {
    if (c.getPoints().size() > minVerticalEdgeLength) {
      approximations.add(c.getPolygonApproximation());
    }
  }

  println("num approximations: " + approximations.size());

  // find the rightmost approximation
  float frameRight = 0;
  for (Contour edge : approximations) {
    float edgeX = edge.getPoints().get(0).x;

    if (edgeX > frameRight && edgeX < searchColumn) {
      frameRight = edgeX;
    }
  }

  float frameLeft = frameRight - frameWidth;

  // === CALCULATE THE FRAME LOCATION and EXTRACT IT ===

  // make a rectangle starting at frameLeft and topSprocket.top
  // and extending the width and height of the frame
  selectedArea = new Rectangle((int)frameLeft, topSprocketEdge, frameWidth, frameHeight);
  // create a PImage for output and copy
  // the pixels from the source image into it.
  // NB: have to adjust the y-position down by the y-position of the ROI
  output = createImage(selectedArea.width, selectedArea.height, ARGB);
  output.copy(src, selectedArea.x, roi.y + selectedArea.y, selectedArea.width, selectedArea.height, 0, 0, selectedArea.width, selectedArea.height);
}

void drawContours(ArrayList<Contour> cntrs) {
  for (Contour contour : cntrs) {
    contour.draw();
  }
}

void draw() {
  background(125);
  fill(0);
  text("press 's' to save output frame", 563, 756);
  scale(0.5);
  image(src, 0, 0);

  pushMatrix();
  scale(0.75);
  image(dst, src.width * 4.0/3, 0);
  image(dst2, src.width * 4.0/3, roi.height + 10);
  popMatrix();

  image(output, src.width + 10, roi.height + 400);
  noFill();
  strokeWeight(4);

  stroke(255, 0, 0);
  rect(roi.x, roi.y, roi.width, roi.height);
  strokeWeight(1);

  stroke(0, 0, 255);
  translate(roi.x, roi.y);
  line(searchColumn, 0, searchColumn, src.height);

  stroke(255);
  strokeWeight(3);
  drawContours(approximations);

  strokeWeight(5);
  stroke(0, 255, 0);
  rect(selectedArea.x, selectedArea.y, selectedArea.width, selectedArea.height);
}

void keyPressed() {
  if (key == 's') {
    output.save("output/out" + inputFilename);
  }
}

