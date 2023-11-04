package PTransE;

import java.io.*;
import java.util.*;

import static PTransE.GlobalValue.*;
import static PTransE.Gradient.calc_sum;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
public class Test {
    // region private members
    private List<Integer> fb_h; //测试集三元组的head
    private List<Integer> fb_l; //测试集三元组的relation
    private List<Integer> fb_r; //测试集三元组的tail
    private Map<Pair<Integer, Integer>, Set<Integer>> head_relation2tail; // to save the (h, r, t) in train set
    // endregion

    Test() {
        fb_h = new ArrayList<>();
        fb_l = new ArrayList<>();
        fb_r = new ArrayList<>();
        head_relation2tail = new HashMap<>();
    }

    public void add(int head, int relation, int tail, boolean flag) {
        /**
         * head_relation2tail用于存放 正确的三元组
         * flag=true 表示该三元组关系正确
         */
        if (flag) {
            Pair<Integer, Integer> key = new Pair<>(head, relation);
            if (!head_relation2tail.containsKey(key)) {
                head_relation2tail.put(key, new HashSet<>());
            }
            Set<Integer> tail_set = head_relation2tail.get(key);
            tail_set.add(tail);
        } else {
            fb_h.add(head);
            fb_r.add(relation);
            fb_l.add(tail);
        }
    }

    private void Read_Vec_File(String file_name, double[][] vec) throws IOException {
        File f = new File(file_name);
        BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(f),"UTF-8"));
        String line;
        for (int i = 0; (line = reader.readLine()) != null; i++) {
            String[] line_split = line.split("\t");
            for (int j = 0; j < vector_len; j++) {
                vec[i][j] = Double.valueOf(line_split[j]);
            }
        }
    }

    private void relation_add(Map<Integer, Integer> relation_num, int relation) {
        if (!relation_num.containsKey(relation)) {
            relation_num.put(relation, 0);
        }
        int count = relation_num.get(relation);
        relation_num.put(relation, count + 1);
    }

    private void map_add_value(Map<Integer, Integer> tmp_map, int id, int value) {
        if (!tmp_map.containsKey(id)) {
            tmp_map.put(id, 0);
        }
        int tmp_value = tmp_map.get(id);
        tmp_map.put(id, tmp_value + value);
    }

    //返回是否已经存在或者不符合
    //true: 不符合或者已经存在于训练集中
    //false: 符合并且不存在于训练集中
    private boolean hrt_isvalid(int head, int relation, int tail) {
        /**
         * 如果实体之间已经存在正确关系，则不需要计算距离
         * 如果头实体与尾实体一致，也排除该关系的距离计算
         */
        if (head == tail) {
            return true;
        }
        Pair<Integer, Integer> key = new Pair<>(head, relation);
        Set<Integer> values = head_relation2tail.get(key);
        if (values == null || !values.contains(tail)) {
            return false;
        } else {
            return true;
        }
    }

    public void run() throws IOException {
        relation_vec = new double[relation_num][vector_len];
        entity_vec = new double[entity_num][vector_len];

        Read_Vec_File("resource/result/relation2vec.txt", relation_vec);
        Read_Vec_File("resource/result/entity2vec.txt", entity_vec);

        int lsum = 0, rsum = 0;
        int lp_n = 0, rp_n = 0;
        Map<Integer, Integer> lsum_r = new HashMap<>();
        Map<Integer, Integer> rsum_r = new HashMap<>();
        Map<Integer, Integer> lp_n_r = new HashMap<>();
        Map<Integer, Integer> rp_n_r = new HashMap<>();
        Map<Integer, Integer> rel_num = new HashMap<>();

        // File out_file = new File("resource/result/output_detail.txt");
        // OutputStreamWriter writer = new OutputStreamWriter(new FileOutputStream(out_file), "UTF-8");


        //关系预测的一些指标

        int relation_pos_pos = 0;
        int relation_pos_neg = 0;
        int relation_neg_pos = 0;
        int relation_neg_neg = 0;

        System.out.printf("Total iterations = %s\n", fb_l.size());
        String filePath="output.txt";
        try (PrintWriter writer = new PrintWriter(new FileWriter(filePath))) {
            for (int id = 0; id < fb_l.size(); id++) {

                int head = fb_h.get(id);
                int tail = fb_l.get(id);
                int relation = fb_r.get(id);
    //            System.out.println(String.valueOf(id) + ": " + String.valueOf(head) + " " + String.valueOf(tail) + " " + String.valueOf(relation));

                relation_add(rel_num, relation);
                //预测关系
                if(relation != expolitRelation && relation != notExpolitRelation)
                {
                    System.out.println(String.valueOf(head) + " " + String.valueOf(tail) + " " + String.valueOf(relation));
                    continue;
                }
                if(relation == expolitRelation)
                {
                    int neg_relation = notExpolitRelation;
                    if(!hrt_isvalid(head, relation, tail) && !hrt_isvalid(head, neg_relation, tail))
                    {
                        // double predict_score=calc_sum(tail, relation, neg_relation)-calc_sum(tail, relation, relation);
                        double predict_score=1/(1+Math.exp(calc_sum(tail, relation, neg_relation)-calc_sum(tail, relation, relation)));
                        writer.println(predict_score+"\t"+1);
                    }
                }
                else
                {
                    int neg_relation = expolitRelation;
                    if(!hrt_isvalid(head, relation, tail) && !hrt_isvalid(head, neg_relation, tail))
                    {
                        // double predict_score=calc_sum(tail, relation, relation)-calc_sum(tail, relation, neg_relation);
                        double predict_score=1/(1+Math.exp(calc_sum(tail, relation, relation)-calc_sum(tail, relation, neg_relation)));
                        writer.println(predict_score+"\t"+0);
                    }
                }
            }
        }
    }
}
