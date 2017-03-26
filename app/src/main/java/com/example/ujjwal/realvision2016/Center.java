package com.example.ujjwal.realvision2016;

/**
 * Created by Ujjwal on 1/8/17.
 */

public class Center {
    public double x, y;

    public Center(double _x, double _y){
        this.x = _x;
        this.y = _y;
    }

    @Override
    public boolean equals(Object c2){
        if (c2 instanceof Center) {
            return ((Center) c2).x == this.x && ((Center) c2).y == this.y;
        }
        else{
            return false;
        }
    }
}
