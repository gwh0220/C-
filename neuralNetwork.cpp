/***************
//C语言神经网络编程 
//作者：gwh 
//时间：2022.6.28 
***************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

const int inputnodes=784;//设置输入层节点个数 
const int hiddennodes=200;//设置隐藏层节点个数 
const int outputnodes=10;//设置输出层节点个数 
double learningrate=0.1;//设置学习率 
int epochs=3;//设置学习轮次数 

char train_file[100]="C:\\Users\\gwh_0\\Desktop\\mnist_dataset\\mnist_train.csv";//设置训练文件地址 
char test_file[100]="C:\\Users\\gwh_0\\Desktop\\mnist_dataset\\mnist_test.csv";//设置测试文件地址 

double wih[hiddennodes][inputnodes];
double who[outputnodes][hiddennodes];
int iepochs=0;

//生成权重矩阵，并将矩阵的值赋为-0.5~0.5之间 
void matrix_init(){
	for (int i = 0; i < hiddennodes; i++)
	    {
	        for (int j = 0; j < inputnodes; j++)
	        {
	            srand((unsigned)time(NULL)+(unsigned)rand());//重新生成种子 
				wih[i][j]=((rand()%32768)/32767.0)-0.5;
//				printf("%f ",wih[i][j]);
	        }
	    }
	    for (int i = 0; i < outputnodes; i++)
	    {
	        for (int j = 0; j < hiddennodes; j++)
	        {
	            srand((unsigned)time(NULL)+(unsigned)rand());//重新生成种子 
				who[i][j]=((rand()%32768)/32767.0)-0.5;
//				printf("%f ",who[i][j]);
	        }
	    }
}

//神经网络训练函数，传入训练文件地址，函数将完成网络训练 
void train(char *filename){
	FILE *fp = fopen(filename,"r");
	int line=10000;//定义一行最大的字符数 
	char c[line];
	
//	统计csv文件行数
	long file_row=0;
	fseek(fp,0,SEEK_SET);
	while(fgets(c,line,fp) != NULL){
//	    printf("%s", c);
	    file_row++;
	    }
//	printf("行数 = %d\n",file_row);
	
//	统计csv文件列数
	long file_column=1;
	fseek(fp,0,SEEK_SET);
	fgets(c,line,fp);
	char *p=strstr(c,",");
	while((p=strstr(p,","))!=NULL){
		file_column++;
		p++;
	}
//	printf("列数 = %d\n",file_column);

//	训练神经网络 
	int target;
	char *temp=NULL;
	fseek(fp,0,SEEK_SET);
	double input_list[inputnodes][1],target_list[outputnodes][1];
	
	printf("学习率：%.2f\n",learningrate);
	printf("第%d轮，共%d轮 ",iepochs+1,epochs);
	printf("进度：[%.2lf%%]\n",0/file_row);

	
	for(int i=0,r=0;i<file_row;i++){
//	    输出进度 
		if((r%(file_row/10)) == 0){
			system("cls");
			printf("学习率：%.2f\n",learningrate);
			for(int i=0;i<iepochs;i++){
				printf("第%d轮 ",i+1);
	    		printf("进度：[%.2lf%%]\n",100.0);
			}
			printf("第%d轮，共%d轮 ",iepochs+1,epochs); 
	    	printf("进度：[%.2lf%%]\n",r*100.0/file_row);
	    }
	    r++;
	    fgets(c,line,fp);
	    p=c;
//	    printf("%s",p);
	    temp=strstr(p,",");
		*temp=0;
		target=atof(p);
//		printf("%d ",target);
		for(int j=0;j<outputnodes;j++){
			target_list[j][0]=0.01;
		}
		target_list[target][0]=0.99;
		p+=strlen(p)+1;
		for(int k=0;k<file_column-1;k++){
	   		temp=strstr(p,",");
			if(temp!=NULL){
				*temp=0;
//				将数据转换为0.01到1的范围 
				input_list[k][0]=(atof(p)/255.0*0.99)+0.01;
				p+=strlen(p)+1;
			}
			else{
				input_list[k][0]=(atof(p)/255.0*0.99)+0.01;
			}
		}
		
		double hidden_outputs[hiddennodes][1],final_outputs[outputnodes][1];
		double output_errors[outputnodes][1],hidden_errors[hiddennodes][1];
		double whoT[hiddennodes][outputnodes];
//		hidden_outputs[200,1]
	    for(int i=0;i<hiddennodes;i++){
	        for(int j=0;j<1;j++){
	            hidden_outputs[i][j]=0;
	            for(int k=0;k<inputnodes;k++){
	                hidden_outputs[i][j]+=wih[i][k] * input_list[k][j];
	            }
	        }
	    }
	    for(int i=0;i<hiddennodes;i++){
	    	for(int j=0;j<1;j++){
	    		hidden_outputs[i][j]=(1.0/(1+exp(-hidden_outputs[i][j])));
			}
		}
//		final_outputs[10,1]
		for(int i=0;i<outputnodes;i++){
	        for(int j=0;j<1;j++){
	            final_outputs[i][j]=0;
	            for(int k=0;k<hiddennodes;k++){
	                final_outputs[i][j]+=who[i][k] * hidden_outputs[k][j];
	            }
	        }
	    }
	    for(int i=0;i<outputnodes;i++){
	    	for(int j=0;j<1;j++){
	    		final_outputs[i][j]=(1.0/(1+exp(-final_outputs[i][j])));
			}
		}
//		output_errors[10,1]
		for(int i=0;i<outputnodes;i++){
			for(int j=0;j<1;j++){
				output_errors[i][j]=target_list[i][j]-final_outputs[i][j];
			}
		}
//		who转置 [200,10]
		for(int i=0;i<hiddennodes;i++){
	    	for(int j=0;j<outputnodes;j++){
	    		whoT[i][j]=who[j][i];
			}
		}
//		hidden_errors[200,1]
		for(int i=0;i<hiddennodes;i++){
	        for(int j=0;j<1;j++){
	            hidden_errors[i][j]=0;
	            for(int k=0;k<outputnodes;k++){
	                hidden_errors[i][j]+=whoT[i][k] * output_errors[k][j];
	            }
	        }
	    }
	    
//	    更新权重矩阵who 
	    double m1[outputnodes][1];
	    double hidden_outputsT[1][200];
	    for(int i=0;i<outputnodes;i++){
	    	for(int j=0;j<1;j++){
	    		m1[i][j]=output_errors[i][j] * final_outputs[i][j] * (1.0 - final_outputs[i][j]);
			}
		}
		
		for(int i=0;i<1;i++){
	    	for(int j=0;j<hiddennodes;j++){
	    		hidden_outputsT[i][j]=hidden_outputs[j][i];
			}
		}
		for(int i=0;i<outputnodes;i++){
	        for(int j=0;j<hiddennodes;j++){
	            for(int k=0;k<1;k++){
	                who[i][j]+=learningrate*(m1[i][k] * hidden_outputsT[k][j]);
	            }
	        }
	    }
	    
//	    更新权重矩阵wih
		double m2[hiddennodes][1];
	    double input_listT[1][784];
	    for(int i=0;i<hiddennodes;i++){
	    	for(int j=0;j<1;j++){
	    		m2[i][j]=hidden_errors[i][j] * hidden_outputs[i][j] * (1.0 - hidden_outputs[i][j]);
			}
		}
		for(int i=0;i<1;i++){
	    	for(int j=0;j<inputnodes;j++){
	    		input_listT[i][j]=input_list[j][i];
			}
		}
		for(int i=0;i<hiddennodes;i++){
	        for(int j=0;j<inputnodes;j++){
	            for(int k=0;k<1;k++){
	                wih[i][j]+=learningrate*(m2[i][k] * input_listT[k][j]);
	            }
	        }
	    }
	}
	system("cls");
	printf("学习率：%.2f\n",learningrate);
	for(int i=0;i<iepochs+1;i++){
		printf("第%d轮 ",i+1);
		printf("进度：[%.2lf%%]\n",100.0);
	}
	
	fclose(fp);
}

//根据样本数据正向计算网络，得到神经网络正确率 
void query(char *filename){
	FILE *fp = fopen(filename,"r");
	int line=10000;//定义一行最大的字符数 
	char c[line];
	
//	统计csv文件行数 
	long file_row=0;
	fseek(fp,0,SEEK_SET);
	while(fgets(c,line,fp) != NULL){
//	    printf("%s", c);
	    file_row++;
	    }
//	printf("行数 = %d\n",file_row);
	
//	统计csv文件列数 
	long file_column=1;
	fseek(fp,0,SEEK_SET);
	fgets(c,line,fp);
	char *p=strstr(c,",");
	while((p=strstr(p,","))!=NULL){
		file_column++;
		p++;
	}
//	printf("列数 = %d\n",file_column);

//	正向计算神经网络 
	int target;
	char *temp=NULL;
	fseek(fp,0,SEEK_SET);
	double input_list[inputnodes][1],target_list[outputnodes][1];
	int score=0;

	for(int i=0;i<file_row;i++){
	    fgets(c,line,fp);
	    p=c;
//	    printf("%s",p);
	    temp=strstr(p,",");
		*temp=0;
		target=atof(p);
//		printf("%d ",target);
		p+=strlen(p)+1;
		for(int k=0;k<file_column-1;k++){
	   		temp=strstr(p,",");
			if(temp!=NULL){
				*temp=0;
//				将数据转换为0.01到1的范围 
				input_list[k][0]=(atof(p)/255.0*0.99)+0.01;
				p+=strlen(p)+1;
			}
			else{
				input_list[k][0]=(atof(p)/255.0*0.99)+0.01;
			}
		}
		
		double hidden_outputs[hiddennodes][1],final_outputs[outputnodes][1];
		double output_errors[outputnodes][1],hidden_errors[hiddennodes][1];
		double whoT[hiddennodes][outputnodes];
//		hidden_outputs[200,1]
	    for(int i=0;i<hiddennodes;i++){
	        for(int j=0;j<1;j++){
	            hidden_outputs[i][j]=0;
	            for(int k=0;k<inputnodes;k++){
	                hidden_outputs[i][j]+=wih[i][k] * input_list[k][j];
	            }
	        }
	    }
	    for(int i=0;i<hiddennodes;i++){
	    	for(int j=0;j<1;j++){
	    		hidden_outputs[i][j]=(1.0/(1+exp(-hidden_outputs[i][j])));
			}
		}
//		final_outputs[10,1]
		for(int i=0;i<outputnodes;i++){
	        for(int j=0;j<1;j++){
	            final_outputs[i][j]=0;
	            for(int k=0;k<hiddennodes;k++){
	                final_outputs[i][j]+=who[i][k] * hidden_outputs[k][j];
	            }
	        }
	    }
	    for(int i=0;i<outputnodes;i++){
	    	for(int j=0;j<1;j++){
	    		final_outputs[i][j]=(1.0/(1+exp(-final_outputs[i][j])));
			}
		}
		double max=-1;
		int mi=-1;
		for(int i=0;i<outputnodes;i++){
//			printf("%f ",final_outputs[i][0]);
		    if(final_outputs[i][0]>max){
		    	max=final_outputs[i][0];
		    	mi=i;
			}
		}
//		printf("%d  ",mi);
//		printf("\n");
		if(mi==target){
			score++;
		}
	}
	double performance=0;
	performance=score*1.0/file_row;
	printf("正确率:%.2lf%%",performance*100.0);
	
	fclose(fp);
}

int main(){
	matrix_init();
	
	for(iepochs=0;iepochs<epochs;iepochs++){
		train(train_file);
	}
	
	query(test_file);
	
	return 0;
}
