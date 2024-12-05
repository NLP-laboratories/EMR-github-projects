import json
from collections import defaultdict

def count_nested_entities_with_totals(json_data):
    # 初始化嵌套实体统计字典
    nested_entity_count = defaultdict(lambda: defaultdict(int))
    total_nested_count = defaultdict(int)

    for entry in json_data:
        labels = entry.get('label', {})
        entity_positions = []

        # 收集所有实体的位置信息
        for entity_type, entities in labels.items():
            for entity, positions in entities.items():
                for pos in positions:
                    entity_positions.append((entity_type, pos, entity))

        # 对实体位置进行排序，以便处理嵌套关系
        entity_positions.sort(key=lambda x: (x[1][0], -x[1][1]))

        # 统计嵌套关系
        for i in range(len(entity_positions)):
            outer_type, outer_pos, outer_entity = entity_positions[i]

            for j in range(i + 1, len(entity_positions)):
                inner_type, inner_pos, inner_entity = entity_positions[j]

                # 检查是否嵌套
                if (outer_pos[0] <= inner_pos[0] and outer_pos[1] >= inner_pos[1]):
                    nested_entity_count[outer_type][inner_type] += 1
                    total_nested_count[inner_type] += 1

    return nested_entity_count, total_nested_count

def main():
    # 加载JSON数据
    with open('/home/deng/Maping/Nested_NER_experiment/GlobalPointer_pytorch/datasets/CHD/1.json', 'r', encoding='utf-8') as file:
        json_data = [json.loads(line) for line in file]

    # 统计嵌套实体数量
    nested_entity_count, total_nested_count = count_nested_entities_with_totals(json_data)

    # 输出每个实体类别的嵌套实体及其数量
    for outer_type, nested_info in nested_entity_count.items():
        print(f"实体类别: {outer_type}")
        for inner_type, count in nested_info.items():
            print(f"  包含的嵌套实体类别: {inner_type}, 数量: {count}")
    
    # 输出每个嵌套实体类别的总数量
    print("\n总的嵌套实体类别数量：")
    for inner_type, total_count in total_nested_count.items():
        print(f"嵌套实体类别: {inner_type}, 总数量: {total_count}")

if __name__ == "__main__":
    main()
