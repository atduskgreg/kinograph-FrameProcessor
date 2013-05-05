import gab.opencvpro.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.Core;
import org.opencv.core.Mat;

import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;

import org.opencv.core.Point;

import java.util.Arrays;


OpenCVPro opencv;
PImage src, dst;

ArrayList<MatOfPoint> contours;
ArrayList<MatOfPoint2f> approximations;

float medianAngle = 0;

Point[] medianLine;

void setup() {
  src = loadImage("_MG_1264.JPG");
  src.resize(500, 0);
  size(src.width*2, src.height);
  opencv = new OpenCVPro(this, src.width, src.height);

  opencv.loadImage(src);
  opencv.gray();

  opencv.findSobelEdges(2, 0);
  opencv.threshold(150);

  contours = new ArrayList<MatOfPoint>();
  Imgproc.findContours(opencv.getBufferGray(), contours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_NONE);

  approximations = createPolygonApproximations(filterContours(contours));

  float[] angles = new float[approximations.size()];
  int i = 0;
  for (MatOfPoint2f approx : approximations) {
    Point[] points = approx.toArray();

    PVector p1 = new PVector((float)points[0].x, (float)points[0].y);
    PVector p2 = new PVector((float)points[1].x, (float)points[1].y);

    PVector l = PVector.sub(p1, p2);
    PVector vert = new PVector(0, 1);

    float a = PVector.angleBetween(l, vert);
    angles[i] = a;
    i++;
  }

  medianLine = approximations.get(floor(angles.length/2)).toArray();

  Arrays.sort(angles);
  println(angles);
  medianAngle = angles[floor(angles.length/2)];
  medianAngle = radians(180 - degrees(medianAngle));

  println("median angle: " + medianAngle);

  dst = createImage(src.width, src.height, ARGB);
  opencv.toPImage(opencv.getBufferGray(), dst);
}

ArrayList<MatOfPoint> filterContours(ArrayList<MatOfPoint> cntrs) {
  ArrayList<MatOfPoint> result = new ArrayList<MatOfPoint>();
  for (MatOfPoint contour : cntrs) {
    if (contour.toArray().length > 200) {
      result.add(contour);
    }
  }
  return result;
}


void drawContours(ArrayList<MatOfPoint2f> cntrs) {
  for (MatOfPoint2f contour : cntrs) {
    beginShape();
    Point[] points = contour.toArray();

    for (int i = 0; i < points.length; i++) {
      vertex((float)points[i].x, (float)points[i].y);
    }
    endShape(CLOSE);
  }
}

ArrayList<MatOfPoint2f> createPolygonApproximations(ArrayList<MatOfPoint> cntrs) {
  ArrayList<MatOfPoint2f> result = new ArrayList<MatOfPoint2f>();

  double epsilon = cntrs.get(0).size().height * 0.01;
  println(epsilon);

  for (MatOfPoint contour : cntrs) {
    MatOfPoint2f approx = new MatOfPoint2f();
    Imgproc.approxPolyDP(new MatOfPoint2f(contour.toArray()), approx, epsilon, true);
    result.add(approx);
  }

  return result;
}

void draw() {
  pushMatrix();
  
  rotate(medianAngle);
  
  image(src, 0,0);
  image(dst, src.width,0);
  
  stroke(0,255,0);
  strokeWeight(1);
  translate(src.width, 0);
  drawContours(approximations);
  
  popMatrix();
}

