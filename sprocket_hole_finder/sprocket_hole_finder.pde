import gab.opencvpro.*;
import java.awt.Rectangle;

String inputFilename = "_MG_1267.JPG";
PImage src;
OpenCVPro opencv;

ArrayList<Contour> contours;

int roiLeft = 445;
int roiRight = 495;

int minPoints = 30;

void setup() {
  src = loadImage(inputFilename);
  src.resize(0, 800);
  size(src.width, src.height);

  opencv = new OpenCVPro(this, src);
  opencv.setROI(roiLeft, 0, roiRight - roiLeft, height);
  opencv.equalizeHistogram();
  opencv.findSobelEdges(0, 2);
  opencv.threshold(50);
  opencv.dilate();
  opencv.erode();
  opencv.dilate();
  opencv.erode();

  contours = opencv.findContours();
}

void draw() {
  image(opencv.getOutput(), 0, 0);
  stroke(255, 0, 0);
  for (Contour c : contours) {
    if (c.numPoints() > minPoints) {
      float y = c.getPolygonApproximation().getBoundingBox().y;
      line(0,y,width,y);
    }
  }
}
