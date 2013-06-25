import gab.opencvpro.*;
import java.awt.Rectangle;

String inputFilename = "_MG_1267.JPG";
PImage src;
OpenCVPro opencv;

int roiLeft, roiRight;
ArrayList<Contour> contours;

int originalSpacer = 2256; // sized out at 1000px tall
float spacerArea;

void setup() {
  src = loadImage(inputFilename);
  src.resize(0, 800);

  float scalingFactor = src.height / 1000.0;  
  spacerArea = originalSpacer * scalingFactor;

  size(src.width, src.height);

  roiLeft = width/2;
  roiRight = roiLeft + 50;

  opencv = new OpenCVPro(this, src);
  opencv.setROI(roiLeft, 0, roiRight-roiLeft, height);
  opencv.invert();
  opencv.threshold(240);
  contours = opencv.findContours();
  noLoop();
}

void draw() {
  image(opencv.getOutput(), 0, 0);

  stroke(255, 0, 0);

  ArrayList<Rectangle> separators = new ArrayList<Rectangle>();

  pushMatrix();
  translate(roiLeft, 0);
  for (Contour contour : contours) {
    Rectangle bb = contour.getBoundingBox();
    if (abs((bb.width * bb.height) - spacerArea) < 50) {
      separators.add(bb);
      rect(bb.x, bb.y, bb.width, bb.height);
    }
  }
  popMatrix();
  
  stroke(0,255,0);
  strokeWeight(2);

  Rectangle bottom = separators.get(0);
  Rectangle top = separators.get(1);

  line(0, bottom.y, width, bottom.y);
  line(0, top.y + top.height, width, top.y + top.height);
}

