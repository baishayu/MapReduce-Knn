package edu.zzti.knn;

import org.apache.commons.beanutils.BeanUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.io.*;
import java.lang.reflect.InvocationTargetException;
import java.util.*;

/**
 * Created by wanglei on 2018/4/21.
 */

public class KnnDesign {

    private static final int K = 50;
    private static final List<Float []> testData = new ArrayList();

    static class TrainDistanceMapper extends Mapper<LongWritable, Text, LongWritable, Tuple> {

        /**
         * 读取测试集数据  封装到 ArrayList里面
         * @param context
         * @throws IOException
         * @throws InterruptedException
         */

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {

            try {
                //Configuration conf = new Configuration();
                //URI uri  = new URI("hdfs://hadoop01:9000");
                //FileSystem fs  = FileSystem.get(uri,conf,"hadoop");

                FileInputStream inputStream = new FileInputStream("E:\\hdfs_test\\irisdata\\input\\test\\test.txt");
                BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream));
                String line = null;
                while (((line = reader.readLine())!=null)){
                    String[] strs = line.split(",");
                    Float[] features = new Float[strs.length];
                    for (int i = 0; i < features.length; i++) {
                         features[i] = new Float(strs[i]);
                    }
                    testData.add(features);
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }


        /**
         *
         * 读取训练集数据
         * 输出 key是训练集数据的编号
         *      value是特征标签和到该特征标签的距离
         * @param key
         * @param value
         * @param context
         * @throws IOException
         * @throws InterruptedException
         */
        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {

            String[] strs = value.toString().split(",");
            //将strs 分成两部分 一部分是特征值向量  一个是特征标签
            //特征标签
            String mark = strs[strs.length-1];
            //特征值向量
            //为了和测试数据一致  这里空出来 第0位数据
            Float[] trainFeatures = new Float[strs.length];
            for (int i = 1; i < trainFeatures.length; i++) {
                trainFeatures[i] = new Float(strs[i-1]);
            }

            //求距离 并且输出
            for (Float [] testFeatures : testData) {
                Float dist = getDistance(trainFeatures,testFeatures);
                //输出 每一个测试数据的编号  和 到该训练数据的<mark,距离>
                Tuple tuple = new Tuple(mark, dist);
                //System.out.println("mapper:"+tuple);
                context.write(new LongWritable(testFeatures[0].longValue()),tuple);
            }
        }


        /**
         * 求两个向量之间的欧式距离
         * 数组的第0位不参与运算
         * @param trainFeatures
         * @param testFeatures
         * @return
         */
        private Float getDistance(Float[] trainFeatures, Float[] testFeatures) {
            //求差平方 和
            float sum = 0;
            for (int i = 1; i < trainFeatures.length; i++) {
                   sum += Math.pow(testFeatures[i]-trainFeatures[i],2);
            }
            //求开根号
            return new Float(Math.sqrt(sum));
        }


    }

    /**
     * 根据局部最优是全局最优
     * 局部取K值
     */
    static class TrainKCombiner extends Reducer<LongWritable,Tuple,LongWritable,Tuple>{
        @Override
        protected void reduce(LongWritable key, Iterable<Tuple> values, Context context) throws IOException, InterruptedException {
            List<Tuple> tuples = new ArrayList<>();
            Iterator<Tuple> iterator = values.iterator();
            while (iterator.hasNext()){
                //必须使用copy 不能使用引用
                try {
                    Tuple tuple = new Tuple();
                    BeanUtils.copyProperties(tuple,iterator.next());
                    tuples.add(tuple);
                } catch (IllegalAccessException e) {
                    e.printStackTrace();
                } catch (InvocationTargetException e) {
                    e.printStackTrace();
                }
            }

            //如果样本本身就小于 K个直接输出
            if (tuples.size()<=K){
                for (Tuple tuple : tuples) {
                    context.write(key,tuple);
                }
                return;
            }

            //如果样本大于K个值的话 降序排序 取K值
            //按照tuple的距离降序排序
            Collections.sort(tuples);
            //排序后
            System.out.println("排序后"+key+":----");
            System.out.println(tuples);

            //取k值
            for (int i = 0; i < K; i++) {
                System.out.println(key+" combiner:"+tuples.get(i));
                context.write(key,tuples.get(i));
            }

        }
    }

    /**
     * 全局取K值
     */
    static class TrainKReducder extends Reducer<LongWritable,Tuple,LongWritable,Text>{



        @Override
        protected void reduce(LongWritable key, Iterable<Tuple> values, Context context) throws IOException, InterruptedException {

            //用于单词计数的map集合
            Map<String,Integer> wordcount = new TreeMap<>();

            List<Tuple> tuples = new ArrayList<>();
            Iterator<Tuple> iterator = values.iterator();
            while (iterator.hasNext()) {
                //必须使用copy 不能使用引用
                try {
                    Tuple tuple = new Tuple();
                    BeanUtils.copyProperties(tuple, iterator.next());
                    tuples.add(tuple);
                } catch (IllegalAccessException e) {
                    e.printStackTrace();
                } catch (InvocationTargetException e) {
                    e.printStackTrace();
                }
            }
            if(tuples.size() > K){
                //如果样本大于K个值的话 降序排序 取K值
                //按照tuple的距离降序排序
                Collections.sort(tuples);
                List list = new ArrayList();
                //取k值
                for (int i = 0; i < K; i++) {
                    list.add(tuples.get(i));
                }
                tuples = list;
            }
            System.out.println(key+" reducer:"+tuples );

            //进行wordcount 统计
            for (Tuple tuple : tuples) {
                Integer value = wordcount.get(tuple.getMark());
                wordcount.put(tuple.getMark(),value == null ? 1 : value+1 );
            }

            System.out.println("key :"+wordcount);

            Integer maxCount = 0;
            String label = "";
            //求最大值
            for (Map.Entry<String, Integer> entry : wordcount.entrySet()) {
                if(maxCount < entry.getValue()){
                    maxCount = entry.getValue();
                    label = entry.getKey();
                }
            }
            context.write(key,new Text(label));
        }
    }

    public static void main(String[] args) throws IOException, ClassNotFoundException, InterruptedException {

        Configuration conf = new Configuration();
        //conf.set("mapreduce.framework.name","yarn");
        //conf.set("yarn.resourcemanager.hostname","hdpmaster");

        Job job = Job.getInstance(conf);
        //指定本程序的所在路径
        job.setJarByClass(KnnDesign.class);

        //指定本业务job要使用的mapper /reduce业务类
        job.setMapperClass(KnnDesign.TrainDistanceMapper.class);
        job.setCombinerClass(KnnDesign.TrainKCombiner.class);
        job.setReducerClass(KnnDesign.TrainKReducder.class);

        job.setMapOutputKeyClass(LongWritable.class);
        job.setMapOutputValueClass(Tuple.class);

        //指定最终输出的数据的kv类型
        job.setOutputKeyClass(LongWritable.class);
        job.setOutputValueClass(Text.class);

        //指定job的输入原始文件所在目录
        FileInputFormat.setInputPaths(job, "E:\\hdfs_test\\irisdata\\input\\train\\irisdata.txt");

        //输出job的输出原始文件所在目录
        FileOutputFormat.setOutputPath(job,new Path("E:\\hdfs_test\\irisdata\\output"));
        //将job中配置的相关参数，以及job所用的java类所在的jar包，提交给yarn去运行

//		job.submit();
        boolean res = job.waitForCompletion(true);
        System.out.println(res);
        System.exit(res?0:1);


    }

}
