(defproject clj-predict-pth "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "EPL-2.0 OR GPL-2.0-or-later WITH Classpath-exception-2.0"
            :url "https://www.eclipse.org/legal/epl-2.0/"}
  :dependencies [[org.clojure/clojure "1.12.0"]
                 [org.clojure/data.json "2.5.1"]
                 [ai.djl/api "0.31.1"]
                 ;;ai.djl.pytorch:pytorch-native-cpu:1.11.0:linux-x86_64 + ai.djl.pytorch:pytorch-jni
                 [ai.djl.pytorch/pytorch-engine "0.31.1"]
                 [ai.djl.pytorch/pytorch-native-cpu "2.5.1" :classifier "linux-x86_64"]
                 [ai.djl.pytorch/pytorch-jni "2.5.1-0.31.1"]
                 [ai.djl/model-zoo "0.31.1"]]
  :repl-options {:init-ns clj-predict-pth.core})
