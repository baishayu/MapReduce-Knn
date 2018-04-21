package edu.zzti.knn;

import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableComparable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

public class Tuple implements WritableComparable<Tuple>{

    //标签
    private String mark;
    //距离
    private float dist;

    public Tuple() { }

    public Tuple(String mark, Float dist) {
        this.mark = mark;
        this.dist = dist;
    }

    public String getMark() {
        return mark;
    }

    public void setMark(String mark) {
        this.mark = mark;
    }

    public float getDist() {
        return dist;
    }

    public void setDist(float dist) {
        this.dist = dist;
    }

    @Override
    public void write(DataOutput out) throws IOException {
        out.writeUTF(this.mark);
        out.writeFloat(this.dist);
    }

    @Override
    public void readFields(DataInput in) throws IOException {
        this.mark = in.readUTF();
        this.dist = in.readFloat();
    }



    @Override
    public String toString() {
        return "[mark:"+mark + ",dist=" + dist +"]";
    }

    //从小到大排序 表示 距离越近 越是一类
    @Override
    public int compareTo(Tuple o) {
        return Float.compare(this.getDist(),o.getDist());
    }
}